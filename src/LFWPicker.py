import os
import shutil
import random
import tarfile
import urllib.request


class LFWPicker:
    def __init__(self, dataset_path=None, output_path=".", data_url="http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"):
        """
        Initialize the LFWPicker with the path to the LFW dataset.

        :param dataset_path: Path to the LFW dataset.
        :param output_path: Path to the output directory where folders will be created.
        :param data_url: URL to download the LFW dataset.
        """
        self.output_path = output_path
        self.data_url = data_url
        self.dataset_path = dataset_path or self._download_and_extract_dataset()
        self.identities = self._get_identities()

    def _download_and_extract_dataset(self):
        """
        Download and extract the LFW dataset.

        :return: Path to the extracted LFW dataset.
        """
        data_dir = os.path.join(self.output_path, "../data")
        os.makedirs(data_dir, exist_ok=True)
        tgz_path = os.path.join(data_dir, "lfw-funneled.tgz")
        dataset_path = os.path.join(data_dir, "lfw_funneled")

        if not os.path.exists(dataset_path):
            print("Downloading the LFW dataset...")
            urllib.request.urlretrieve(self.data_url, tgz_path)
            print("Extracting the LFW dataset...")
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            os.remove(tgz_path)

        return dataset_path

    def _get_identities(self):
        """
        Get all identities from the dataset.

        :return: A list of all identities in the dataset.
        """
        identities = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        return identities

    def pick_identities(self, N):
        """
        Randomly pick N identities from the dataset.

        :param N: Number of identities to pick.
        :return: A list of N randomly picked identities.
        """
        if N > len(self.identities):
            raise ValueError(f"Requested {N} identities, but only {len(self.identities)} are available in the dataset.")
        return random.sample(self.identities, N)

    def copy_identities(self, picked_identities, destination_folder):
        """
        Copy the picked identities and their images to the destination folder.

        :param picked_identities: List of picked identities.
        :param destination_folder: Destination folder to copy the identities to.
        """
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
        os.makedirs(destination_folder)

        for identity in picked_identities:
            src = os.path.join(self.dataset_path, identity)
            dst = os.path.join(destination_folder, identity)
            shutil.copytree(src, dst)

    def generate_folder_with_identities(self, N):
        """
        Pick N identities and generate a folder with the selected identities.

        :param N: Number of identities to pick.
        """
        picked_identities = self.pick_identities(N)
        destination_folder = os.path.join(self.output_path, f'I-{N}')
        self.copy_identities(picked_identities, destination_folder)
        print(f'{N} identities have been copied to the folder: {destination_folder}')


# Example usage:

output_path = '../data/'
picker = LFWPicker(dataset_path=None, output_path=output_path)  # use the dataset_path='path to existing lfw' parameter to use an
# existing lfw instead of downloading.
for k in [50, 100, 200, 500, 1000]:
    picker.generate_folder_with_identities(k)  # generate the test datasets I-K.