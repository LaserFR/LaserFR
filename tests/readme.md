# 1. Celebrities Recognition

This class utilizes the Amazon Rekognition service to recognize celebrities in images stored in a specified directory.

## Usage

1. **Set Up AWS Credentials**:
   - Ensure you have your AWS credentials configured. You can do this by setting up the `~/.aws/credentials` file or using environment variables.

2. **Directory Structure**:
   - Place your images in the `data/real_images` directory.

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python celebrities_recognition.py
     ```
   - This will process all images in the directory, if one attacker is recognized as a celebrity, the output will be '{attacker name} is recognized as {celebrity name} with confidence score {confidence score}'. And the total results will be saved in `results/celeb_results.csv`.

## Example Usage

- **`if __name__ == '__main__':`**:
  - Sets the path to the directory containing images (`data/real_images`).
  - Instantiates the `CelebritiesRecognition` class with the specified path.
  - Calls the `main` method to start the celebrity recognition process.

## Important Notes

- **AWS Rekognition Costs**:
  - Be aware that using Amazon Rekognition incurs costs. Check AWS pricing for details.
  
- **Error Handling**:
  - The current implementation does not include error handling to manage issues like network errors, unsupported file formats, etc. We will add them in the later version.

# 2. Continual Attack

This class uses TensorFlow and the FaceNet model to perform continual attack tests. It can train a classifier, classify images, and iteratively classify and retrain a model on successfully impersonated attacker images.

## Usage

1. **Set Up Environment**:
   
   - The original version of Facenet is built with TensorFlow V1.x. The code has been adjusted to compile with the new version of TensorFlow.
     
   - Make sure the pre-trained FaceNet model [20180402-114759.pb](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) is placed in the Models folder.

3. **Directory Structure**:
   
   - The targeted dataset is the `lfw_funneled` saved in the `data` directory.
     
   - The attacker dataset is the `/data/real_images`

4. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python continual_attack.py
     ```

## Example of How to Use the Code

- **`if __name__ == '__main__':`**:
  - Creates an instance of the `ContinualAttack` class with the specified parameters.
  - Calls the `train` method to train the classifier.
  - Calls the `classify` method to classify images using the trained classifier.
  - Calls the `classify_attackers_and_retrain` method to classify and retrain on attacker images.
  - The retraining rounds and the attackers that have been classified as the initial target are printed each round until all attackers are classified.

## Important Notes

- **FaceNet Model**:
  Since FaceNet is built on TensorFlow 1.x, there may be numerous compatibility issues. Therefore, we recommend running it on a CPU instead of a GPU. And we modified the original `facenet.py` to solve the compatibility issues.

# 3. DodgingDoS

This class uses the MTCNN and InceptionResnetV1 models from the `facenet-pytorch` library to analyze face embeddings and detect attempts at dodging or Denial-of-Service (DoS) attacks using synthetic images.

## Usage

2. **Directory Structure**:
   - Place your original attacker images in the `original_data_dir` directory (e.g., data/attackers).
   - Place your synthetic attacker images in the `synthetic_data_dir` directory (e.g., data/synthetic_attackers).
   - Place the images in denylist in the `db_data_dir` directory (e.g., data/I-100).

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python dodgings.py
     ```
   - the final results are saved to `results/id_dodging.csv` and `results/db_dodging.csv` for identity dodging and database dodging, respectively.

## Example of How to Use the Code

- **`if __name__ == '__main__':`**:
  - Specifies the directories for original and synthetic attacker images and the denylist.
  - Creates an instance of the `DodgingDoS` class with the specified directories.
  - Calls the `analyze_attackers` method to analyze the images and get the results.
  - Saves the results to a CSV file using the `save_results` method.

- **Threshold Value**:
  - The threshold value for detecting dodging attempts is set to 1.242 by default, which is from the [FaceNet paper](https://arxiv.org/abs/1503.03832) (tested with LFW). Adjust this value as needed based on your requirements.


# 4. Real-Time Face Recognition with deepface

This script uses the DeepFace library to perform real-time face recognition using a webcam. The script allows you to specify the model, distance metric, and other parameters for the recognition process.

## Usage

1. **Install DeepFace**:
   - Ensure you have the DeepFace library installed. You can install it via pip:
     ```bash
     pip install deepface
     ```
2. **Replace the realtime.py file**
    Find the realtime.py file in the 'common' folder in the deepface, and replace it with our realtime.py file to support embedding saving and loading, and a more natural face recognition approach.
   
2. **Prepare the Database**:
   - Place the face database in the directory specified by db_path.

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python real_timeFR.py
     ```
   - This will start the real-time face recognition stream using your default webcam.

## Example of How to Use the Code

- **`if __name__ == '__main__':`**:
  - `db_path` (str): Path to the directory containing the database of known faces.
  - `model_name` (str): Name of the model to use for face recognition (e.g., 'Facenet').
  - `distance_metric` (str): Distance metric to use for face comparison (e.g., 'euclidean_l2').
  - `enable_face_analysis` (bool): Whether to enable additional face analysis (age, gender, emotion) (default is `False`).
  - `detector_backend` (str): Face detector backend to use (e.g., 'mtcnn').

## Important Notes

### Embedding Data
- The embedding data will be saved as `embedding.pkl` in the `db_path` directory. This allows the data to be loaded directly in the future, eliminating the need for recalculation.
  But for a new data path, the embeddings need to be recalculated and saved.

### Face Detection and Analysis
- The script uses the MTCNN backend for face detection and the specified model for face recognition. You can test with different backends, FR models, and distance metrics supported by [deepface](https://github.com/serengil/deepface). Additional face analysis (age, gender, emotion) can be enabled if required.

### Webcam Source
- The script uses the default webcam (source `0`). You can change the source if you have multiple webcams or other video sources.



