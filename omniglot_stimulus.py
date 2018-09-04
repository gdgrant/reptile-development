import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import os


class Stimulus:

    def __init__(self):

        self.num_training_languages = 30
        self.num_testing_languages = 20
        self.num_character_samples = 20

        self.training_characters = self.get_character_paths('./omniglot/images_background/')
        self.testing_characters = self.get_character_paths('./omniglot/images_evaluation/')


    def get_character_paths(self, dir):

        paths = []
        for l in os.listdir(dir):
            for c in os.listdir(dir+l):
                paths.append(dir+l+'/'+c)

        return paths


    def make_task(self, num_samples=None, eval=False):
        if num_samples is None:
            num_samples = self.num_character_samples

        chars = self.testing_characters if eval else self.training_characters
        self.task_chars = np.random.choice(chars, size=par['n_ways'], replace=False)

        self.train_samples = []
        self.test_samples = []
        for n in range(par['n_ways']):
            sample_set = np.random.choice(self.num_character_samples, \
                size=num_samples, replace=False)

            if eval:
                train_set = sample_set[:par['n_shots']]
                test_set  = sample_set[par['n_shots']:]
            else:
                train_set = sample_set[:num_samples]
                test_set  = sample_set[num_samples:]

            self.train_samples.append(train_set)
            self.test_samples.append(test_set)

        self.used_train_samples = [[] for n in range(len(self.task_chars))]
        self.rotation = np.random.choice(4, size=len(self.task_chars))


    def make_batch(self, batch_size, test=False, eval=False):

        batch_data = np.zeros([batch_size, 26, 26], dtype=np.float32)
        batch_labels = np.zeros([batch_size, par['n_output']], dtype=np.float32)
        samples = self.test_samples if test else self.train_samples

        for b in range(batch_size):
            if test or eval:
                char_index   = np.random.choice(len(self.task_chars))
                sample_index = np.random.choice(samples[char_index])
            else:
                while True:
                    valid_chars = [c for c in range(len(self.task_chars)) if (len(self.used_train_samples[c]) < self.num_character_samples)]
                    char_index   = np.random.choice(valid_chars)
                    sample_index = np.random.choice(samples[char_index])
                    if sample_index not in self.used_train_samples[char_index]:
                        self.used_train_samples[char_index].append(sample_index)
                        break

            s = os.listdir(self.task_chars[char_index])[sample_index]
            p = self.task_chars[char_index]+'/'+s

            base = 1 - plt.imread(p)
            base = np.rot90(base, k=self.rotation[char_index])
            for i in range(2):
                base = (base[:-1:2,:-1:2]+base[1::2,1::2])/2

            batch_data[b,:,:] = base
            batch_labels[b,char_index] = 1

        return batch_data, batch_labels, -1


if __name__ == '__main__':
    s = Stimulus()
    s.make_task()
    for i in range(5):
        print('BATCH')
        s.make_batch(par['pre_train_batch_size'])

    print(s.used_train_samples)
