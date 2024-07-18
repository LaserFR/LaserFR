import os
import pandas as pd
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class AttackInformer:
    def __init__(self, model_name, metric_name, backend_name, targets_path, attacker_path):
        # Initialize the AttackInformer with model details and paths
        self.model_name = model_name
        self.metric_name = metric_name
        self.backend_name = backend_name
        self.targets_path = targets_path
        self.attacker_path = attacker_path

    @staticmethod
    def find_image(base_path, name, valid_extensions=('.jpg', '.jpeg', '.png')):
        # Find an image file with the given name in the base_path
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.startswith(name) and file.lower().endswith(valid_extensions):
                    return os.path.join(root, file)
        return None

    @staticmethod
    def extract_attacker_and_laser(identity):
        # Extract attacker name and laser setting from the identity string
        parts = identity.replace('\\', '/').rsplit('/', 2)
        attacker_name = parts[1].split('_')[0]
        laser_setting = parts[-1].rsplit('_', 1)[-1].split('.')[0]
        return attacker_name, laser_setting

    def perform_face_recognition(self, img_path, db_path):
        # Perform face recognition using DeepFace
        return DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=self.model_name,
            distance_metric=self.metric_name,
            detector_backend=self.backend_name,
            enforce_detection=False
        )

    def process_attacker_image(self, attacker_img_path, target_name=None):
        # Process a single attacker image and return the results
        df_result = self.perform_face_recognition(attacker_img_path, self.targets_path)

        if df_result is not None and not df_result.empty:
            # Extract attacker information and add it to the results
            attacker_name, laser_setting = self.extract_attacker_and_laser(attacker_img_path)
            df_result['attacker'] = attacker_name
            df_result['laser setting'] = laser_setting
            df_result['identity'] = df_result['identity'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

            if target_name:
                # If a target name is provided, check for matches
                df_result['target'] = target_name
                if target_name in df_result['identity'].values:
                    matching_row = df_result[df_result['identity'] == target_name]
                    lowest_metric_row = df_result.loc[df_result[f'{self.model_name}_{self.metric_name}'].idxmin()]
                    return df_result, matching_row, lowest_metric_row

            return df_result

        return None

    def inform_untargeted_attack(self):
        # Perform untargeted attack analysis
        results = []
        configurations = f"{self.model_name}_{self.metric_name}_{self.backend_name}"

        with ThreadPoolExecutor() as executor:
            for attacker_name in os.listdir(self.attacker_path):
                attacker_folder = os.path.join(self.attacker_path, attacker_name)
                if os.path.isdir(attacker_folder):
                    # Process all images in the attacker folder
                    attacker_images = [os.path.join(attacker_folder, img) for img in os.listdir(attacker_folder) if
                                       os.path.isfile(os.path.join(attacker_folder, img))]
                    print(f"Running attack with configuration: {configurations}")
                    future_results = executor.map(self.process_attacker_image, attacker_images)
                    results.extend(filter(lambda x: x is not None, future_results))

        if results:
            # Combine and save results
            total_results = pd.concat(results, ignore_index=True)
            results_path = os.path.join("..", "results", f"{configurations}_untargeted_impersonation_results.csv")
            total_results.to_csv(results_path, index=False)
            print(f"Untargeted impersonation completed and results saved in {results_path}")
        else:
            print("No results for untargeted attack found.")

    def inform_targeted_attack(self):
        # Perform targeted attack analysis
        csv_path = '../results/selected_pairs_1000.csv'
        df_pairs = pd.read_csv(csv_path)

        results, results_matching, results_lowest = [], [], []

        with ThreadPoolExecutor() as executor:
            # Process each target-attacker pair in parallel
            future_to_pair = {executor.submit(self.process_pair, row['Target'], row['Attacker']): row for _, row in
                              df_pairs.iterrows()}
            for future in future_to_pair:
                pair_results = future.result()
                if pair_results:
                    results.extend(pair_results[0])
                    results_matching.extend(pair_results[1])
                    results_lowest.extend(pair_results[2])

        # Save different types of results
        self.save_results(results, "new_targeted_impersonation_all_results.csv")
        self.save_results(results_matching, "new_targeted_impersonation_matching_results.csv")
        self.save_results(results_lowest, "new_targeted_impersonation_lowest_results.csv")

    def process_pair(self, target_name, attacker_name):
        # Process a single target-attacker pair
        attacker_folder = os.path.join(self.attacker_path, attacker_name)
        if os.path.isdir(attacker_folder):
            attacker_images = [os.path.join(attacker_folder, img) for img in os.listdir(attacker_folder) if
                               self.is_image(os.path.join(attacker_folder, img))]
            results = [self.process_attacker_image(img, target_name) for img in attacker_images]
            return [r for r in results if r is not None]
        return None

    def save_results(self, results, filename):
        # Save results to a CSV file
        if results:
            final_results_df = pd.concat(results, ignore_index=True)
            results_path = os.path.join("..", "results", f"{self.model_name}_{filename}")
            final_results_df.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
        else:
            print(f"No results found for {filename}")

    @staticmethod
    def is_image(file_path):
        # Check if a file is an image based on its extension
        return os.path.isfile(file_path) and file_path.lower().endswith(('.jpg', '.jpeg', '.png'))


if __name__ == '__main__':
    # Set up parameters and paths
    models = ["DeepFace", "ArcFace", "SFace"] #"VGG-Face",
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    backends = ['mtcnn', 'opencv']

    psas_selected_images_path = os.path.normpath(os.path.join('..', 'selected_data', 'psas_selected_images'))
    synthetic_attackers_path = os.path.normpath(os.path.join('..', 'data', 'synthetic_attackers'))
    targets_base_path = 'E:/lfw_funneled_random/croppedrandom_100'

    # Create AttackInformer instance and run untargeted attack

    for model in models:
        for metric in metrics:
            for backend in backends:
                # Create AttackInformer instance
                informer = AttackInformer(model, metric, backend, targets_base_path, synthetic_attackers_path)
                # Run the targeted attack
                informer.inform_untargeted_attack()

