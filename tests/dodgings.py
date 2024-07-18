from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pandas as pd
import os
from PIL import Image


def calculate_distance(embedding1, embedding2):
    return (embedding1 - embedding2).norm().item()


def compare_embeddings(original_embeddings, synthetic_embeddings, threshold):
    results = []
    for synthetic_img, synthetic_embedding in synthetic_embeddings:
        for img_file, original_embedding in original_embeddings:
            distance = calculate_distance(original_embedding, synthetic_embedding)
            print(f'distance between {img_file} and {synthetic_img} is {distance} ')
            if distance > threshold:
                results.append((img_file, synthetic_img, distance, synthetic_embedding))
                print(f'{synthetic_img} succeed in dodging')
    return results


def save_inconsistencies(inconsistencies, output_file):
    df = pd.DataFrame(inconsistencies, columns=['failed  identity', 'Synthetic Image'])
    df.to_csv(output_file, index=False, header=True)


def save_results(results, output_file):
    df = pd.DataFrame(results, columns=['Attacker', 'Synthetic Image', 'Distance'])
    df.to_csv(output_file, index=False, header=True)


class DodgingDoS:
    def __init__(self, original_data_dir, synthetic_data_dir, deny_data_dir, device=None):
        self.original_data_dir = original_data_dir
        self.synthetic_data_dir = synthetic_data_dir
        self.deny_data_dir = deny_data_dir  # Directory containing the 1k data for comparison
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Initialize MTCNN (for face detection) and InceptionResnetV1 (for face embedding extraction)
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                           post_process=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.results = []  # This will store the results of analyze_attackers
        print(f'Running on device: {self.device}')

    def get_embedding(self, img_path):
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None
        img_cropped = self.mtcnn(img)
        if img_cropped is None:
            return None
        img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device)).detach().cpu()
        return img_embedding

    def dodging(self, threshold=1.242):
        self.results = []
        for attacker in os.listdir(self.original_data_dir):
            attacker_dir = os.path.join(self.original_data_dir, attacker)
            synthetic_attacker_path = os.path.join(self.synthetic_data_dir, attacker)
            original_embeddings = self.process_directory(attacker_dir)
            synthetic_embeddings = self.process_directory(synthetic_attacker_path)
            self.results.extend(compare_embeddings(original_embeddings, synthetic_embeddings, threshold))
        return self.results

    def process_directory(self, directory_path):
        # List of supported image file extensions
        valid_extensions = ['.jpg', '.jpeg', '.png']  # Add or remove extensions based on your needs
        embeddings = []

        for img_file in os.listdir(directory_path):
            # Check if the file is an image based on its extension
            if any(img_file.lower().endswith(ext) for ext in valid_extensions):
                img_path = os.path.join(directory_path, img_file)
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    embeddings.append((img_file, embedding))
                else:
                    print(f'Image {img_file} succeed in DoS')
            # else:
            #     print(f'Ignored non-image file: {img_file}')
        return embeddings

    def db_dodging(self, threshold=1.242):
        if not self.results:
            print("identity dodging does not succeed. Please make sure again.")
            return []
        db_embeddings = self.process_directory(self.deny_data_dir)

        for result in self.results:
            for db_img, db_embedding in db_embeddings:
                distance = calculate_distance(db_embedding, result[3])  # Compare db embedding with synthetic embedding
                if distance <= threshold:
                    inconsistencies.append((db_img, result[1]))  # Record pairs where the distance condition fails
        return inconsistencies


# Example usage
if __name__ == '__main__':
    original_data_dir = '../data/attackers'
    synthetic_data_dir = '../data/synthetic_attackers'
    db_data_dir = 'E:\lfw_funneled_random\croppedrandom_100'
    analyzer = DodgingDoS(original_data_dir, synthetic_data_dir, db_data_dir)
    results = analyzer.dodging()
    save_results(results, '../results/id_dodging.csv')
    inconsistencies = analyzer.db_dodging()
    save_inconsistencies(inconsistencies, '../results/db_dodging.csv')
