import cv2
import numpy as np
import torch
from torch.autograd import Variable
from Models import resnet152
from deepface.commons import distance
import shutil
import os
from torch.utils.data import Dataset, DataLoader
from mtcnn import MTCNN


class ImageDataset(Dataset):
    def __init__(self, root_folder):
        self.image_paths = []
        # Verify the directory exists
        if not os.path.isdir(root_folder):
            print(f"Directory does not exist: {root_folder}")
            return

        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith(('jpg', 'png')):
                    full_path = os.path.join(subdir, file)
                    self.image_paths.append(full_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image, image_show = self.read(image_path)
        image = cv2.resize(image, (112, 112))
        image = np.transpose(image, (2, 0, 1)).copy()
        image = torch.from_numpy(image.astype(np.float32).copy())
        return image, image_show, image_path

    def read(self, path):
        image = cv2.imread(path).astype(np.float32) / 255.
        image_show = image[:, :, ::-1].copy()
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return image, image_show


class ExplanationGenerator:

    def __init__(self):
        self.Decomposition = None
        self.mtcnn = MTCNN()
        self.model = resnet152()
        resume = 'Models/parameters.pth'
        resume = os.path.join('..', resume)
        print('load model from {}'.format(resume))
        weight = torch.load(resume, map_location=torch.device('cpu'))
        self.model.load_state_dict(weight)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    # read image
    def get_input_from_path(self, folder_1, folder_2, size=(160, 160)):
        print("Initializing datasets...")  # Debug: Print when initializing datasets
        folder_1_path = os.path.join('..', folder_1)  # Adjust path relative to src
        folder_2_path = os.path.join('..', folder_2)  # Adjust path relative to src
        dataset_1 = ImageDataset(folder_1_path)
        dataset_2 = ImageDataset(folder_2_path)
        print(f"Attacker Dataset length: {len(dataset_1)}")  # Debug: Print length of dataset 1
        print(f"Target Dataset length: {len(dataset_2)}")  # Deb
        loader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False)
        loader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False)

        print("Datasets initialized.")
        return loader_1, loader_2

    # calculate embedding
    def get_embed(self, inputs):

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                embed, map_ = self.model(inputs, True)
            else:
                embed, map_ = self.model(inputs, True)
            fc = self.model.fc.cpu()
            bn = self.model.bn3.cpu()

        return embed, map_, fc, bn

    def Overall_map(self, map_1, map_2, fc_1=None, fc_2=None, bn_1=None, bn_2=None, size=(112, 112), mode='Flatten'):
        '''
            From Paper: Only for Flatten architecture, you may check the code of other applications
            for the implementation of GAP and GMP.
        '''
        global Decomposition
        if mode == 'Flatten':

            map_1 = np.transpose(map_1.detach().cpu().numpy(), (0, 2, 3, 1))
            map_2 = np.transpose(map_2.detach().cpu().numpy(), (0, 2, 3, 1))

            map_1_reshape = np.reshape(map_1, [-1, map_1.shape[-1]])
            map_2_reshape = np.reshape(map_2, [-1, map_2.shape[-1]])
            map_1_embed = np.zeros([map_1_reshape.shape[0], fc_1.weight.data.numpy().shape[0]])
            map_2_embed = np.zeros([map_2_reshape.shape[0], fc_2.weight.data.numpy().shape[0]])

            # consider all operations as one linear transformation, compute the equivalent feature for each position
            weight_1 = 1
            bias_1 = 0
            weight_2 = 1
            bias_2 = 0

            if fc_1 is not None and fc_2 is not None:
                weight_1 *= np.reshape(fc_1.weight.data.numpy(),
                                       [fc_1.weight.data.numpy().shape[0], map_1_reshape.shape[-1],
                                        map_1_reshape.shape[0]])
                bias_1 += fc_1.bias.data.numpy() / map_1_reshape.shape[0] / map_1_reshape.shape[1]

                weight_2 *= np.reshape(fc_2.weight.data.numpy(),
                                       [fc_2.weight.data.numpy().shape[0], map_2_reshape.shape[-1],
                                        map_2_reshape.shape[0]])
                bias_2 += fc_2.bias.data.numpy() / map_1_reshape.shape[0] / map_1_reshape.shape[1]

            if bn_1 is not None and bn_2 is not None:
                weight_1 /= np.sqrt(bn_1.running_var.data.numpy())[:, np.newaxis, np.newaxis]
                bias_1 = (bias_1 - bn_1.running_mean.data.numpy()) / np.sqrt(bn_1.running_var.data.numpy())

                weight_1 *= (bn_1.weight.data.numpy())[:, np.newaxis, np.newaxis]
                bias_1 = bias_1 * bn_1.weight.data.numpy() + bn_1.bias.data.numpy()

                weight_2 /= np.sqrt(bn_2.running_var.data.numpy())[:, np.newaxis, np.newaxis]
                bias_2 = (bias_2 - bn_2.running_mean.data.numpy()) / np.sqrt(bn_2.running_var.data.numpy())

                weight_2 *= (bn_2.weight.data.numpy())[:, np.newaxis, np.newaxis]
                bias_2 = bias_2 * bn_2.weight.data.numpy() + bn_2.bias.data.numpy()
            # compute the transformed feature, break apart to avoid too large matrix operation in Memory
            for i in range(map_1_reshape.shape[0]):
                map_1_embed[i] = np.matmul(map_1_reshape[i], np.transpose(weight_1[:, :, i]))  # + bias_1
                map_2_embed[i] = np.matmul(map_2_reshape[i], np.transpose(weight_2[:, :, i]))  # + bias_2

            # reshape back
            map_1_embed = np.reshape(map_1_embed, [map_1.shape[1], map_1.shape[2], -1])
            map_2_embed = np.reshape(map_2_embed, [map_2.shape[1], map_2.shape[2], -1])

            Decomposition = np.zeros([map_1.shape[1], map_1.shape[2], map_2.shape[1], map_2.shape[2]])
            for i in range(map_1.shape[1]):
                for j in range(map_1.shape[2]):
                    for x in range(map_2.shape[1]):
                        for y in range(map_2.shape[2]):
                            Decomposition[i, j, x, y] = np.sum(map_1_embed[i, j] * map_2_embed[x, y])
            Decomposition = Decomposition / np.max(Decomposition)
            Decomposition = np.maximum(Decomposition, 0)
        return Decomposition

    def get_landmarks(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.mtcnn.detect_faces(image_rgb)
        landmarks = result[0]['keypoints']
        return landmarks

    def calculate_a(self, landmarks):
        '''
            calculate the division coordinates, which is the mid of nose and mouth.
        '''
        if landmarks is not None and len(landmarks) > 0:
            nose_y = landmarks['nose'][1]
            mouth_y = (landmarks['left_eye'][1] + landmarks['right_eye'][1]) / 2
            a = int((nose_y + mouth_y) / 2)
        else:
            a = 80  # default value
        return a

    def es_filter(self, folder_1, folder_2, size=(112, 112), metrics='euclidean_l2', move=False):
        '''
            Calculate the distances between each pair of images in the two folders,
            and select the top k shortest distances.
            The default value for k is 25%, but you can change it manually.
        '''
        k = 0.25
        loader_1, loader_2 = self.get_input_from_path(folder_1, folder_2, size=size)

        all_pairs = []
        for batch_1 in loader_1:
            for batch_2 in loader_2:
                inputs_1, image_1, path_1 = batch_1
                inputs_2, image_2, path_2 = batch_2

                embed_1, map_1, fc_1, bn_1 = self.get_embed(inputs_1)
                embed_2, map_2, fc_2, bn_2 = self.get_embed(inputs_2)
                # Detach the tensors before calling numpy()
                embed_1_np = embed_1.detach().cpu().numpy()
                embed_2_np = embed_2.detach().cpu().numpy()

                if metrics == 'euclidean':
                    dist = distance.findEuclideanDistance(embed_1_np, embed_2_np)
                elif metrics == 'euclidean_l2':
                    dist = distance.findEuclideanDistance(distance.l2_normalize(embed_1_np), distance.l2_normalize(embed_2_np))
                elif metrics == 'cosine':
                    dist = distance.findCosineDistance(embed_1_np, embed_2_np)
                else:
                    raise ValueError('Please choose the right metric')

                all_pairs.append((dist, path_1[0], path_2[0], embed_1, map_1, bn_1, fc_1, embed_2, map_2, bn_2, fc_2))
                del inputs_1, inputs_2, embed_1, embed_2, map_1, map_2
                torch.cuda.empty_cache()

            all_pairs.sort(key=lambda x: x[0])  # Sort by distance

            top_k_percent_index = int(len(all_pairs) * k)
            selected_pairs = all_pairs[:top_k_percent_index]
            print(f'{k*100}% targets have been filtered out.')

            if move:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                data_dir = os.path.join(parent_dir, 'selected_data')
                new_folder = os.path.join(data_dir,'./es_selected_images')
                os.makedirs(new_folder, exist_ok=True)
                for _, path_1, path_2, _, _, _, _, _, _, _, _ in selected_pairs:
                    folder_1 = os.path.dirname(path_1)
                    folder_2 = os.path.dirname(path_2)

                    # Extracting the base name of the file in path_2
                    base_name = os.path.splitext(os.path.basename(path_2))[0]

                    new_folder_1 = os.path.join(new_folder, os.path.basename(folder_1))
                    new_folder_2 = os.path.join(new_folder_1, os.path.basename(folder_2))
                    os.makedirs(new_folder_1, exist_ok=True)
                    os.makedirs(new_folder_2, exist_ok=True)

                    # Copy the image from path_2 to new_folder_2
                    new_image_path = os.path.join(new_folder_2, os.path.basename(path_2))
                    shutil.copy2(path_2, new_image_path)
            print(f'selected images have been copied to {new_folder_1}.')

        return selected_pairs

    def psas_filter(self, selected_pairs=None, move=False, size = (112, 112)):

        selected_names = []
        names = []
        for pair in selected_pairs:
            dist, path_1, path_2, embed_1, map_1, bn_1, fc_1, embed_2, map_2, bn_2, fc_2 = pair

            landmarks = self.get_landmarks(path_1)
            a = self.calculate_a(landmarks)

            # Calculate the overall map
            decomposition = self.Overall_map(map_1=map_1, map_2=map_2, fc_1=fc_1, fc_2=fc_2, bn_1=bn_1, bn_2=bn_2)
            decom_1 = cv2.resize(np.sum(decomposition, axis=(2, 3)), (size[1], size[0]))
            decom_1 = decom_1 / np.max(decom_1)

            # Calculate the mean values of the two parts of the decomposition
            upper_part = decom_1[:][:a]
            lower_part = decom_1[:][a:]

            mean_upper = np.mean(upper_part)
            mean_lower = np.mean(lower_part)

            # Check if mean_upper < mean_lower
            if mean_upper < mean_lower:
                selected_names.append((path_1, path_2))
            if move:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                data_dir = os.path.join(parent_dir, 'selected_data')
                new_folder = os.path.join(data_dir, 'psas_selected_images')

                os.makedirs(new_folder, exist_ok=True)
                for path_1, path_2 in selected_names:
                    folder_1 = os.path.dirname(path_1)
                    folder_2 = os.path.dirname(path_2)

                    # Extracting the base name of the file in path_2
                    base_name = os.path.splitext(os.path.basename(path_2))[0]

                    new_folder_1 = os.path.join(new_folder, os.path.basename(folder_1))
                    new_folder_2 = os.path.join(new_folder_1, os.path.basename(folder_2))
                    os.makedirs(new_folder_1, exist_ok=True)
                    os.makedirs(new_folder_2, exist_ok=True)

                    # Copy the image from path_2 to new_folder_2
                    new_image_path = os.path.join(new_folder_2, os.path.basename(path_2))
                    shutil.copy2(path_2, new_image_path)

            # Clear CUDA cache to free up memory

            del embed_1, embed_2, decomposition
            torch.cuda.empty_cache()
        associated_names = {}
        for path_1, path_2 in selected_names:
            name_1 = os.path.basename(path_1)
            name_2 = os.path.basename(path_2)

            if name_1 not in associated_names:
                associated_names[name_1] = set()
            associated_names[name_1].add(name_2)

        print(f'Total unique names: {len(associated_names)}')
        for name_1, name_2_set in associated_names.items():
            print(f'{name_1}: {len(name_2_set)} names -> {name_2_set}')

        return associated_names


if __name__ == '__main__':
    theOne = 'data/theOne'
    theMany = 'data/theMany'

    eg = ExplanationGenerator()
    selected_pairs = eg.es_filter(theOne, theMany, move=True)
    selected_names = eg.psas_filter(selected_pairs, move=True)


