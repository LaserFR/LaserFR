# 1. CelebritiesRecognition Class

This class utilizes the Amazon Rekognition service to recognize celebrities in images stored in a specified directory.

## Usage

1. **Set Up AWS Credentials**:
   - Ensure you have your AWS credentials configured. You can do this by setting up the `~/.aws/credentials` file or using environment variables.

2. **Directory Structure**:
   - Place your images in the `data/real_images` directory.
   - Ensure the `results` directory exists or the code will need to create it before saving the CSV.

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python celebrities_recognition.py
     ```
   - This will process all images in the specified directory, recognize any celebrities, and save the results in `results/celeb_results.csv`.

## Example of How to Use the Code

- **`if __name__ == '__main__':`**:
  - Sets the path to the directory containing images (`data/real_images`).
  - Instantiates the `CelebritiesRecognition` class with the specified path.
  - Calls the `main` method to start the celebrity recognition process.

## Important Notes

- **AWS Rekognition Costs**:
  - Be aware that using Amazon Rekognition incurs costs. Check AWS pricing for details.
  
- **Image Formats**:
  - Ensure your images are in a format supported by Amazon Rekognition (JPEG or PNG).

- **Error Handling**:
  - The current implementation does not include error handling. It is recommended to add error handling to manage issues like network errors, unsupported file formats, etc.

# 2. FaceClassifier Class

This class uses TensorFlow and the FaceNet model to perform continual attack tests. It can train a classifier, classify images, and iteratively classify and retrain a model on successfully impersonated attacker images.

## Usage

1. **Set Up Environment**:
   - Ensure you have TensorFlow, numpy, scikit-learn, and PIL installed.
   - Download and place the FaceNet model in the specified path.

2. **Directory Structure**:
   - Place your images in the `data_dir` directory.
   - Ensure the model path and classifier filename are correctly specified.

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python continual_attack.py
     ```
   - This will train the classifier, classify images, and iteratively classify and retrain on attacker images.

## Example of How to Use the Code

- **`if __name__ == '__main__':`**:
  - Creates an instance of the `FaceClassifier` class with the specified parameters.
  - Calls the `train` method to train the classifier.
  - Calls the `classify` method to classify images using the trained classifier.
  - Calls the `classify_attackers_and_retrain` method to classify and retrain on attacker images.

## Important Notes

- **FaceNet Model**:
  - Ensure you have the pre-trained FaceNet model. Download it if necessary. Since FaceNet is built on TensorFlow 1.x, there may be numerous compatibility issues. Therefore, we recommend running it on a CPU instead of a GPU.

- **Image Formats**:
  - Ensure your images are in a format supported by PIL (JPEG, PNG, etc.).

- **Error Handling**:
  - The current implementation includes basic error handling for image file validation. Additional error handling may be required for robustness.


# 3. DodgingDoS Class

This class uses the MTCNN and InceptionResnetV1 models from the `facenet-pytorch` library to analyze face embeddings and detect attempts at dodging or Denial-of-Service (DoS) attacks using synthetic images.

## Usage

1. **Set Up Environment**:
   - Ensure you have `torch`, `facenet-pytorch`, `numpy`, `pandas`, and `PIL` installed.
   - Ensure your images are in the specified directories.

2. **Directory Structure**:
   - Place your original attacker images in the `original_data_dir` directory.
   - Place your synthetic attacker images in the `synthetic_data_dir` directory.

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python dodging.py
     ```
   - This will analyze the images, detect dodging and DoS attempts, and save the results to `../results/dodgingDoS.csv`.

## Example of How to Use the Code

- **`if __name__ == '__main__':`**:
  - Specifies the directories for original and synthetic attacker images.
  - Creates an instance of the `DodgingDoS` class with the specified directories.
  - Calls the `analyze_attackers` method to analyze the images and get the results.
  - Saves the results to a CSV file using the `save_results` method.

## Important Notes

- **MTCNN and InceptionResnetV1 Models**:
  - Ensure you have the `facenet-pytorch` library installed to use these models.

- **Image Formats**:
  - Ensure your images are in a format supported by PIL (JPEG, PNG, etc.).

- **Threshold Value**:
  - The threshold value for detecting dodging attempts is set to 1.242 by default, which is from the [FaceNet paper](https://arxiv.org/abs/1503.03832) (tested with LFW). Adjust this value as needed based on your requirements.


# 4. Real-Time Face Recognition with DeepFace

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



