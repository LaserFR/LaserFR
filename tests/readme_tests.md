# CelebritiesRecognition Class

This class utilizes the Amazon Rekognition service to recognize celebrities in images stored in a specified directory.

## Initialization
- **`__init__(self, path)`**:
  - Initializes the `CelebritiesRecognition` class.
  - Sets up a client for Amazon Rekognition.
  - Takes a `path` parameter, which is the directory path containing the images.

## Methods

- **`recognize_celebrities(self, photo)`**:
  - Recognizes celebrities in the given photo using Amazon Rekognition.
  - Reads the image file and sends it to the Rekognition `recognize_celebrities` API.
  - Parses the response to extract and print celebrity details such as name, ID, confidence, and related URLs.
  - Returns the number of recognized celebrities and a list containing the photo name, celebrity name, confidence, and URLs.

- **`main(self)`**:
  - Walks through the directory specified by `self.path` to get a list of all image files.
  - Calls the `recognize_celebrities` method for each image file.
  - If any celebrities are detected in an image, the results are appended to a list.
  - Converts the results list to a pandas DataFrame and saves it as a CSV file named `celeb_results.csv` in the `results` directory.

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

