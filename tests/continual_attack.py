import tensorflow as tf
import numpy as np
from facenet.src import facenet
import os
import math
import pickle
from sklearn.svm import SVC
from PIL import Image


class ContinualAttack:
    def __init__(self, data_dir, model, classifier_filename, use_split_dataset=False,
                 test_data_dir=None, mode='TRAIN', batch_size=10, image_size=160, seed=666,
                 min_nrof_images_per_class=10, nrof_train_images_per_class=5):
        '''
        :param data_dir: Directory containing the face images for training.
        :param model: Path to the pre-trained FaceNet model
        :param classifier_filename: Path to save the trained classifier
        :param use_split_dataset: Flag to indicate if the dataset should be split into training and testing sets
        :param test_data_dir: Directory for test data (optional)
        :param mode: Mode of operation, either 'TRAIN' or 'CLASSIFY'
        :param batch_size: Batch size for processing images.
        :param image_size: Size of the images.
        :param seed: Random seed for reproducibility.
        :param min_nrof_images_per_class: Minimum number of images per class.
        :param nrof_train_images_per_class: Number of images per class for training.
        '''
        self.data_dir = data_dir
        self.model = model
        self.classifier_filename = classifier_filename
        self.use_split_dataset = use_split_dataset
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = seed
        self.min_nrof_images_per_class = min_nrof_images_per_class
        self.nrof_train_images_per_class = nrof_train_images_per_class
        self.mode = mode

        self.train_set = None
        self.test_set = None
        self.class_names = None
        self.probability_threshold = None

        tf.compat.v1.disable_eager_execution()

    def is_image_file(self, file_path):
        # Check if the given file path is an image
        try:
            img = Image.open(file_path)
            img.verify()  # Verify that it is, in fact, an image
            return True
        except (IOError, SyntaxError):
            return False

    def load_dataset(self):
        # Load dataset and filter out classes with fewer images than min_nrof_images_per_class
        dataset_tmp = facenet.get_dataset(self.data_dir)
        for cls in dataset_tmp:
            cls.image_paths = [path for path in cls.image_paths if self.is_image_file(path)]
        filtered_dataset = [cls for cls in dataset_tmp if
                            len(cls.image_paths) >= self.min_nrof_images_per_class]

        # Splits the dataset into training and testing sets if `use_split_dataset` is `True`.
        if self.use_split_dataset:
            self.train_set, self.test_set = self.split_dataset(filtered_dataset)
        else:
            self.train_set = filtered_dataset
            self.test_set = filtered_dataset

        for cls in self.train_set:
            assert len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset'

    def split_dataset(self, dataset):
        # Split dataset into training and testing sets
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            if len(paths) >= self.min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(facenet.ImageClass(cls.name, paths[:self.nrof_train_images_per_class]))
                test_set.append(facenet.ImageClass(cls.name, paths[self.nrof_train_images_per_class:]))
        return train_set, test_set

    def calculate_embeddings(self, sess, paths):
        # Calculate embeddings for the the images using the FaceNet model
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

        embedding_size = embeddings.get_shape()[1]

        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        return emb_array

    def train(self):
        # Train the classifier using SVM
        with tf.compat.v1.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.seed)
                self.load_dataset()

                paths, labels = facenet.get_image_paths_and_labels(self.train_set)
                facenet.load_model(self.model)

                emb_array = self.calculate_embeddings(sess, paths)

                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                self.class_names = [cls.name.replace('_', ' ') for cls in self.train_set]

                # Saves the trained classifier to `classifier_filename`.
                with open(self.classifier_filename, 'wb') as outfile:
                    pickle.dump((model, self.class_names), outfile)
                print(f'Saved classifier model to file "{self.classifier_filename}"')

                self.probability_threshold = np.min(model.predict_proba(emb_array))  # Set probability threshold
                print(f'Set probability threshold to {self.probability_threshold}')  # Print threshold

    def classify(self):
        # Classify images using the trained classifier
        with tf.compat.v1.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.seed)
                self.load_dataset()

                paths, labels = facenet.get_image_paths_and_labels(self.test_set)
                facenet.load_model(self.model)

                emb_array = self.calculate_embeddings(sess, paths)

                with open(self.classifier_filename, 'rb') as infile:
                    model, self.class_names = pickle.load(infile)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print(f'{i:4d}  {self.class_names[best_class_indices[i]]}: {best_class_probabilities[i]:.3f}')

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print(f'Accuracy: {accuracy:.3f}')

    def classify_attackers_and_retrain(self, attackers_dir, initial_attacker_index=0):
        # Iteratively classify and retrain the classifier on attacker images
        round_counter = 0
        all_classified = False

        # Continues the process until all attackers are classified.
        while not all_classified and round_counter <= 20:
            round_counter += 1
            with tf.compat.v1.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    np.random.seed(seed=self.seed)
                    self.load_dataset()

                    facenet.load_model(self.model)

                    attackers_dataset = facenet.get_dataset(attackers_dir)
                    for cls in attackers_dataset:
                        cls.image_paths = [path for path in cls.image_paths if self.is_image_file(path)]
                    attacker_paths, attacker_labels = facenet.get_image_paths_and_labels(attackers_dataset)
                    attacker_embs = self.calculate_embeddings(sess, attacker_paths)

                    with open(self.classifier_filename, 'rb') as infile:
                        model, self.class_names = pickle.load(infile)

                    predictions = model.predict_proba(attacker_embs)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    # Track the number of images classified for each attacker subfolder
                    classified_counts = {attacker: 0 for attacker in set(attacker_labels)}

                    # Select one attacker to classify initially
                    initial_attacker_label = best_class_indices[initial_attacker_index]
                    initial_attacker_confidence = best_class_probabilities[initial_attacker_index]

                    if initial_attacker_confidence > self.probability_threshold:
                        print(f'Initial attacker classified as {self.class_names[initial_attacker_label]} '
                              f'with probability {initial_attacker_confidence:.3f}')
                        self.train_set[initial_attacker_label].image_paths.append(
                            attacker_paths[initial_attacker_index])

                        # Retrain the classifier with the initial attacker
                        self.train()

                        # Classify remaining attackers with the new classifier
                        remaining_attacker_paths = attacker_paths[1:]
                        remaining_attacker_embs = attacker_embs[1:]
                        remaining_attacker_labels = attacker_labels[1:]

                        all_classified = True
                        for i in range(len(remaining_attacker_paths)):
                            prediction = model.predict_proba([remaining_attacker_embs[i]])
                            best_class_index = np.argmax(prediction)
                            best_class_probability = prediction[0][best_class_index]

                            if best_class_index == initial_attacker_label and best_class_probability > self.probability_threshold:
                                print(f'Attacker {i + 1} classified as {self.class_names[best_class_index]} '
                                      f'with probability {best_class_probability:.3f}')
                                self.train_set[initial_attacker_label].image_paths.append(remaining_attacker_paths[i])

                                # Update count for the classified attacker
                                classified_counts[remaining_attacker_labels[i]] += 1

                                all_classified = False

                        # Retrain the classifier with the newly classified attackers
                        self.train()

            print(f'Retraining round {round_counter} completed.')
            for attacker_label, count in classified_counts.items():
                print(f'Attacker {attacker_label} has {count} images classified '
                      f'as {self.class_names[initial_attacker_label]}.')

        print(f'All attackers classified in {round_counter} rounds')


if __name__ == '__main__':
    classifier = ContinualAttack(
        data_dir='../data/lfw_funneled',  # Directory for training data
        model='../Models/20180402-114759.pb',   # Path to the pre-trained FaceNet model
        classifier_filename='../Models/classifier.pkl',  # Path to save the trained classifier
        use_split_dataset=False,
        mode='TRAIN'
    )

    # To train the classifier
    classifier.train()

    # To classify using the classifier
    classifier.classify()

    # To classify attackers and retrain the classifier, select one attacker as the initial attacker.
    classifier.classify_attackers_and_retrain('../data/real_images', initial_attacker_index=0)
