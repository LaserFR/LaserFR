import boto3
import json
import os
import pandas as pd


class CelebritiesRecognition:

    def __init__(self, path):
        self.client = boto3.client('rekognition')
        self.path = path

    def recognize_celebrities(self, photo):
        found = []
        with open(photo, 'rb') as image:
            response = self.client.recognize_celebrities(Image={'Bytes': image.read()})

        # print(f'Detected faces for {photo}')

        for celebrity in response['CelebrityFaces']:
            print(f'Name: {celebrity["Name"]}')
            print(f'Id: {celebrity["Id"]}')
            print(f'Confidence: {celebrity["MatchConfidence"]}')
            print('Info:')
            for url in celebrity['Urls']:
                print(f'   {url}')

            found = [os.path.basename(photo), celebrity['Name'], celebrity['MatchConfidence'], celebrity['Urls']]
        return len(response['CelebrityFaces']), found

    def main(self):
        results = []
        filelist = [os.path.join(root, file) for root, dirs, files in os.walk(self.path) for file in files]
        # print(filelist)

        for people in filelist:
            # print(people)
            photo = people
            celeb_count, result = self.recognize_celebrities(photo)
            if celeb_count >= 1:
                print(f'Celebrities detected: {celeb_count}')
                results.append(result)

        df = pd.DataFrame(results, columns=['Photo', 'Name', 'Confidence', 'Urls'])
        results_path = os.path.join("..", "results", "celeb_results.csv")
        df.to_csv(results_path, index=False)


if __name__ == '__main__':
    path = os.path.join("..", "data", "real_images")
    celeb_recognition = CelebritiesRecognition(path)
    celeb_recognition.main()