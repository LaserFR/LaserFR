import boto3
import json
import os
import pandas as pd


class CelebritiesRecognition:

    def __init__(self, path):
        # Initialize the CelebritiesRecognition class
        # Set up the Amazon Rekognition client and the path to the directory containing images
        self.client = boto3.client('rekognition')
        self.path = path

    def recognize_celebrities(self, photo):
        # Recognize celebrities in the given photo using Amazon Rekognition
        found = []

        # Reads the image file and sends it to the Rekognition recognize_celebrities API.
        with open(photo, 'rb') as image:
            response = self.client.recognize_celebrities(Image={'Bytes': image.read()})

        # print(f'Detected faces for {photo}')

        # Parses the response to extract and print celebrity details such as name, ID, confidence, and related URLs
        for celebrity in response['CelebrityFaces']:
            print(f'Name: {celebrity["Name"]}')
            print(f'Id: {celebrity["Id"]}')
            print(f'Confidence: {celebrity["MatchConfidence"]}')
            print('Info:')
            for url in celebrity['Urls']:
                print(f'   {url}')
            # Store the relevant details of the celebrity found
            found = [os.path.basename(photo), celebrity['Name'], celebrity['MatchConfidence'], celebrity['Urls']]

        # Return the number of detected celebrities and the details
        return len(response['CelebrityFaces']), found

    def main(self):
        results = []
        # Get a list of all image files in the directory
        filelist = [os.path.join(root, file) for root, dirs, files in os.walk(self.path) for file in files]
        # print(filelist)

        # Calls the recognize_celebrities method for each image file
        for people in filelist:
            # print(people)
            photo = people
            celeb_count, result = self.recognize_celebrities(photo)
            # If any celebrities are detected in an image, the results are appended to a list.
            if celeb_count >= 1:
                print(f'Celebrities detected: {celeb_count}')
                results.append(result)

        # Converts the results list to a pandas DataFrame,
        # and saves it as a CSV file named celeb_results.csv in the results directory.
        df = pd.DataFrame(results, columns=['Photo', 'Name', 'Confidence', 'Urls'])
        results_path = os.path.join("..", "results", "celeb_results.csv")
        df.to_csv(results_path, index=False)


if __name__ == '__main__':

    # Specify the path to the directory containing the images
    path = os.path.join("..", "data", "real_images")
    # Create an instance of the CelebritiesRecognition class
    celeb_recognition = CelebritiesRecognition(path)
    # Run the main method to start the celebrity recognition process
    celeb_recognition.main()