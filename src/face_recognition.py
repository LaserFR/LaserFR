import time
import pandas as pd
from deepface import DeepFace
import os


def inform_attack(model_name,  metric_name, backend_name, targets_path, attacker_path):
    result_temp = []

    # Iterate over each attacker's folder in attacker_path
    for attacker_name in os.listdir(attacker_path):
        attacker_folder = os.path.join(attacker_path, attacker_name)

        if os.path.isdir(attacker_folder):
            for attacker_img in os.listdir(attacker_folder):
                attacker_img_path = os.path.join(attacker_folder, attacker_img)

                # Ensure it's a file
                if os.path.isfile(attacker_img_path):
                    # Construct the corresponding target path
                    corresponding_target_path = os.path.join(targets_path, attacker_name)

                    if os.path.isdir(corresponding_target_path):
                        # Iterate over each target subfolder within the corresponding target path
                        for target_subfolder in os.listdir(corresponding_target_path):
                            target_subfolder_path = os.path.join(corresponding_target_path, target_subfolder)

                            if os.path.isdir(target_subfolder_path):
                                # face recognition
                                df = DeepFace.find(img_path=attacker_img_path,
                                                   db_path=target_subfolder_path,
                                                   model_name=model_name,
                                                   distance_metric=metric_name,
                                                   detector_backend=backend_name,
                                                   enforce_detection=False)

                                if df is not None and not df.empty:
                                    df['attacker'] = attacker_name
                                    df['laser setting'] = os.path.splitext(attacker_img)[0].split('_')[-1]
                                    df['identity'] = df['identity'].apply(lambda x: os.path.basename(os.path.dirname(x)))
                                    print(df[['attacker', 'laser setting', 'identity']])
                                    result_temp.append(df)

    # Combine all results into a single DataFrame
    if result_temp:
        total_results = pd.concat(result_temp, ignore_index=True)
        results_path = os.path.join("..", "results", "impersonation_results.csv")
        # Save the results to a CSV file
        total_results.to_csv(results_path, index=False)
    else:
        print("No results found.")


if __name__ == '__main__':
    '''
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "SFace"]
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    backends = ['opencv', 'mtcnn', 'retinaface']
    '''
    models = "ArcFace"
    metrics = "euclidean_l2"
    backends = 'mtcnn'

    # Define the paths relative to the current directory
    psas_selected_images_path = os.path.join('..', 'selected_data', 'psas_selected_images')
    synthetic_attackers_path = os.path.join('..', 'data', 'synthetic_attackers')
    # Normalize paths for Windows compatibility
    psas_selected_images_path = os.path.normpath(psas_selected_images_path)
    synthetic_attackers_path = os.path.normpath(synthetic_attackers_path)

    inform_attack(models, metrics, backends, psas_selected_images_path, synthetic_attackers_path)


