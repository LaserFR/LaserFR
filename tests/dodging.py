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
        self.original_data_dir = original_data_dir
        self.synthetic_data_dir = synthetic_data_dir
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
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

    def analyze_attackers(self, threshold=1.424):
        results = []
        for attacker in os.listdir(self.original_data_dir):
            attacker_dir = os.path.join(self.original_data_dir, attacker)

            if not os.path.isdir(attacker_dir):
                print(f'{attacker_dir} is not a directory.')
                continue

            synthetic_attacker_path = os.path.join(self.synthetic_data_dir, attacker)

            if not os.path.isdir(synthetic_attacker_path):
                print(f'Synthetic attacker directory {synthetic_attacker_path} does not exist.')
                continue

            # Get embeddings for all original attacker images
            original_embeddings = []
            for img_file in os.listdir(attacker_dir):
                img_path = os.path.join(attacker_dir, img_file)
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    original_embeddings.append((img_file, embedding))
                else:
                    print(f'Original attacker image {img_file} cannot be detected by MTCNN.')

            # Iterate through synthetic images
            for synthetic_img in os.listdir(synthetic_attacker_path):
                synthetic_img_path = os.path.join(synthetic_attacker_path, synthetic_img)

                # Get synthetic attacker embedding
                synthetic_embedding = self.get_embedding(synthetic_img_path)
                if synthetic_embedding is None:
                    print(f'Synthetic attacker {synthetic_img} succeed in DoS.')
                    results.append((attacker, synthetic_img, 'succeed in DoS', None))
                    continue

                # Compare with all original embeddings
                for img_file, original_embedding in original_embeddings:
                    distance = (original_embedding - synthetic_embedding).norm().item()
                    if distance > threshold:
                        print(f'Synthetic attacker {synthetic_img} succeed in dodging.')
                        results.append((attacker, synthetic_img, 'succeed in dodging', distance))
                        break  # Only need to find one match exceeding the threshold

        return results

    def save_results(self, results, output_file):
        df = pd.DataFrame(results, columns=['Attacker', 'Synthetic Image', 'Detection Status', 'Distance'])
        df.to_csv(output_file, index=False, header=True)


# Example usage
if __name__ == '__main__':
    original_data_dir = '../data/attackers'
    synthetic_data_dir = '../data/synthetic_attackers'
    analyzer = DodgingDoS(original_data_dir, synthetic_data_dir)
    results = analyzer.analyze_attackers(threshold=1.424)
    analyzer.save_results(results, '../results/dodgingDoS.csv')
