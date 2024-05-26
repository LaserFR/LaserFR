# 1. CelebritiesRecognition Class

This class utilizes the Amazon Rekognition service to recognize celebrities in images stored in a specified directory.

## Usage
- **`if __name__ == '__main__':`**:
  - Sets the path to the directory containing images (`data/real_images`).
  - Instantiates the `CelebritiesRecognition` class with the specified path.
  - Calls the `main` method to start the celebrity recognition process.

## Example of How to Use the Code

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

## Important Notes

- **AWS Rekognition Costs**:
  - Be aware that using Amazon Rekognition incurs costs. Check AWS pricing for details.
  
- **Image Formats**:
  - Ensure your images are in a format supported by Amazon Rekognition (JPEG or PNG).

- **Error Handling**:
  - The current implementation does not include error handling. It is recommended to add error handling to manage issues like network errors, unsupported file formats, etc.


# 2. FaceClassifier Class

This class uses TensorFlow and the FaceNet model to perform face recognition and classification. It can train a classifier, classify images, and iteratively classify and retrain a model on attacker images.

## Usage
- **`if __name__ == '__main__':`**:
  - Creates an instance of the `FaceClassifier` class with the specified parameters.
  - Calls the `train` method to train the classifier.
  - Calls the `classify` method to classify images using the trained classifier.
  - Calls the `classify_attackers_and_retrain` method to classify and retrain on attacker images.

## Example of How to Use the Code

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

## Important Notes

- **FaceNet Model**:
  - Ensure you have the pre-trained FaceNet model. Download it if necessary. Since FaceNet is built on TensorFlow 1.x, there may be numerous compatibility issues. Therefore, we recommend running it on a CPU instead of a GPU.

- **Image Formats**:
  - Ensure your images are in a format supported by PIL (JPEG, PNG, etc.).

- **Error Handling**:
  - The current implementation includes basic error handling for image file validation. Additional error handling may be required for robustness.



