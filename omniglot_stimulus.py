import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import os


class Stimulus:

    def __init__(self):

        self.num_training_languages = 30
        self.num_testing_languages = 20

        self.training_languages = os.listdir('./omniglot/images_background/')
        self.testing_languages = os.listdir('./omniglot/images_evaluation/')

        self.select_characters()


    def select_characters(self):

        self.meta_characters = []
        for n in range(par['n_meta_tasks']):
            lang = np.random.choice(self.training_languages)
            char = np.random.choice(os.listdir('./omniglot/images_background/'+lang))
            self.meta_characters.append('./omniglot/images_background/'+lang+'/'+char)

        self.test_characters = []
        for n in range(par['n_test_tasks']*par['testing_repetitions']):
            lang = np.random.choice(self.testing_languages)
            char = np.random.choice(os.listdir('./omniglot/images_evaluation/'+lang))
            self.test_characters.append('./omniglot/images_evaluation/'+lang+'/'+char)


    def generate_omniglot_batch(self, char_path, label, batch_size):

        batch_data = np.zeros([batch_size, 105, 105], dtype=np.float32)
        batch_labels = np.zeros([batch_size, par['n_output']], dtype=np.float32)
        samples = os.listdir(char_path)

        for b in range(batch_size):

            s = np.random.choice(samples)
            batch_data[b,:,:] = 1 - plt.imread(char_path+'/'+s)
            batch_labels[b,label] = 1

        return batch_data, batch_labels


    def make_batch(self, batch_size, task_id, test=False):
        """ Based on the task number and testing status, generate a randomly
            selected or generated batch of images. """

        if test:
            char_path = self.test_characters[task_id]
            label = int(task_id%par['n_test_tasks'] + par['n_meta_tasks'])
        else:
            char_path = self.meta_characters[task_id]
            label = task_id

        batch_data, batch_labels = self.generate_omniglot_batch(char_path, label, batch_size)
        mask = -1

        # Give the images, labels, and mask to the network
        return batch_data, batch_labels, mask
