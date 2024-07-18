from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
from PIL import Image


class DodgingDoS:
    def __init__(self, original_data_dir, synthetic_data_dir, device=None):
        # Initialize DodgingDoS class, set original and synthetic data directories, and configure device (CPU or GPU)
        self.original_data_dir = original_data_dir
        self.synthetic_data_dir = synthetic_data_dir
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Initialize MTCNN (for face detection) and InceptionResnetV1 (for face embedding extraction)
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                           post_process=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print(f'Running on device: {self.device}')

    def get_embedding(self, img_path):
        # Load an image and get its embedding representation
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None

        # Use MTCNN to detect and crop the face
        img_cropped = self.mtcnn(img)
        if img_cropped is None:
            return None

        # Get the embedding of the cropped image
        img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device)).detach().cpu()
        return img_embedding

    def analyze_attackers(self, threshold=1.242):
        # Analyzes the original and synthetic attacker images to detect dodging and DoS attempts.
        results = []
        # Iterates through each attacker in the `original_data_dir`.
        for attacker in os.listdir(self.original_data_dir):
            attacker_dir = os.path.join(self.original_data_dir, attacker)

            if not os.path.isdir(attacker_dir):
                print(f'{attacker_dir} is not a directory.')
                continue

            synthetic_attacker_path = os.path.join(self.synthetic_data_dir, attacker)

            if not os.path.isdir(synthetic_attacker_path):
                print(f'Synthetic attacker directory {synthetic_attacker_path} does not exist.')
                continue

            # For each attacker, gets the embeddings of the original images.
            original_embeddings = []
            for img_file in os.listdir(attacker_dir):
                img_path = os.path.join(attacker_dir, img_file)
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    original_embeddings.append((img_file, embedding))
                else:
                    print(f'Original attacker image {img_file} cannot be detected by MTCNN.')

            # For each synthetic image, compares its embedding with the original embeddings.
            for synthetic_img in os.listdir(synthetic_attacker_path):
                synthetic_img_path = os.path.join(synthetic_attacker_path, synthetic_img)

                # Get synthetic attacker embedding
                synthetic_embedding = self.get_embedding(synthetic_img_path)
                # If no face is detected in the synthetic image, it is considered a successful DoS
                if synthetic_embedding is None:
                    print(f'Synthetic attacker {synthetic_img} succeed in DoS.')
                    results.append((attacker, synthetic_img, 'succeed in DoS', None))
                    continue

                # If the distance between the synthetic and original embeddings exceeds the threshold,
                # it is considered a successful dodging attack.
                for img_file, original_embedding in original_embeddings:
                    distance = (original_embedding - synthetic_embedding).norm().item()
                    if distance > threshold:
                        print(f'Synthetic attacker {synthetic_img} succeed in dodging.')
                        results.append((attacker, synthetic_img, 'succeed in dodging', distance))
                        continue  # To find all matches exceeding the threshold

        return results

    def save_results(self, results, output_file):
        # Save the analysis results to a CSV file
        df = pd.DataFrame(results, columns=['Attacker', 'Synthetic Image', 'Detection Status', 'Distance'])
        df.to_csv(output_file, index=False, header=True)




# Example usage
if __name__ == '__main__':
    original_data_dir = '../data/attackers'  # Directory for original attacker images
    synthetic_data_dir = '../data/synthetic_attackers'  # Directory for synthetic attacker images
    analyzer = DodgingDoS(original_data_dir, synthetic_data_dir)
    results = analyzer.analyze_attackers(threshold=1.242)
    analyzer.save_results(results, '../results/dodgingDoS.csv')
