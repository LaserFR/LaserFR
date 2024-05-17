import tensorflow as tf
import numpy as np
from facenet.src import facenet
import os
import math
import pickle
from sklearn.svm import SVC


class FaceClassifier:
    def __init__(self, data_dir, model, classifier_filename, use_split_dataset=False,
                 test_data_dir=None, mode='TRAIN', batch_size=10, image_size=160, seed=666,
                 min_nrof_images_per_class=1, nrof_train_images_per_class=1):
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

        self.dataset = None
        self.class_names = None

        # Disable eager execution and use TensorFlow 1.x compatibility mode
        tf.compat.v1.disable_eager_execution()

    def load_dataset(self):
        if self.use_split_dataset:
            dataset_tmp = facenet.get_dataset(self.data_dir)
            train_set, test_set = self.split_dataset(dataset_tmp)
            self.dataset = train_set if self.mode == 'TRAIN' else test_set
        else:
            self.dataset = facenet.get_dataset(self.data_dir)

        for cls in self.dataset:
            assert len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset'

    def split_dataset(self, dataset):
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
        with tf.compat.v1.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.seed)
                self.load_dataset()

                paths, labels = facenet.get_image_paths_and_labels(self.dataset)
                facenet.load_model(self.model)

                emb_array = self.calculate_embeddings(sess, paths)

                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                self.class_names = [cls.name.replace('_', ' ') for cls in self.dataset]

                with open(self.classifier_filename, 'wb') as outfile:
                    pickle.dump((model, self.class_names), outfile)
                print(f'Saved classifier model to file "{self.classifier_filename}"')

    def classify(self):
        with tf.compat.v1.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.seed)
                self.load_dataset()

                paths, labels = facenet.get_image_paths_and_labels(self.dataset)
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

    def classify_attackers_and_retrain(self, attackers_dir):
        with tf.compat.v1.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                np.random.seed(seed=self.seed)
                self.load_dataset()

                # Load the model
                facenet.load_model(self.model)

                # Load attacker images
                attackers_dataset = facenet.get_dataset(attackers_dir)
                attacker_paths, _ = facenet.get_image_paths_and_labels(attackers_dataset)
                attacker_embs = self.calculate_embeddings(sess, attacker_paths)

                # Load classifier
                with open(self.classifier_filename, 'rb') as infile:
                    model, self.class_names = pickle.load(infile)

                predictions = model.predict_proba(attacker_embs)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                new_samples = []
                new_labels = []

                for i, (index, probability) in enumerate(zip(best_class_indices, best_class_probabilities)):
                    if probability > 0.012:  # Threshold for considering the classification as confident
                        print(
                            f'Attacker {i} classified as {self.class_names[index]} with probability {probability:.3f}')
                        new_samples.append(attacker_paths[i])
                        new_labels.append(index)
                    else:
                        print(f'Attacker {i} not confidently classified')

                if new_samples:
                    # Add new samples to the dataset
                    for sample, label in zip(new_samples, new_labels):
                        for cls in self.dataset:
                            if cls.name == self.class_names[label]:
                                cls.image_paths.append(sample)

                    # Retrain the classifier
                    self.train()


if __name__ == '__main__':
    # Example usage
    classifier = FaceClassifier(
        data_dir='../data/theMany',
        model='../Models/20180402-114759.pb',
        classifier_filename='../Models/classifier.pkl',
        use_split_dataset=True,
        mode='TRAIN'
    )

    # To train the classifier
    classifier.train()

    # To classify using the classifier
    classifier.classify()

    # To classify attackers and retrain the classifier
    classifier.classify_attackers_and_retrain('../data/synthetic_attackers')
