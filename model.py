# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product
import matplotlib.pyplot as plt
#plt.switch_backend('agg')       # For remote plot generation

# Model modules
from parameters import *
import omniglot_stimulus as stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

    def __init__(self, input_data_batches, output_data_batches, step_size):

        self.input_data = tf.unstack(input_data_batches, axis=0)
        self.output_data = tf.unstack(output_data_batches, axis=0)
        self.step_size = step_size

        self.initialize_variables()
        self.optimize()


    def initialize_variables(self):

        self.var_dict = {}
        for n in range(par['n_layers']-1):
            layer_dict = {}

            with tf.variable_scope('layer'+str(n)):
                W = tf.get_variable('W', initializer=tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], \
                    -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable=True)
                b = tf.get_variable('b', initializer=tf.zeros([1,par['layer_dims'][n+1]]), trainable=True)

            layer_dict['W'] = W
            layer_dict['b'] = b

            self.var_dict['layer'+str(n)] = layer_dict


    def run_model(self, x):

        if par['conv_input']:
            x = self.convolutional(x[...,tf.newaxis])

        x = tf.reshape(x, [-1, par['n_input']])
        y = self.feed_forward(x)

        return y


    def convolutional(self, x):

        for f in [32, 16]:
            conv = tf.layers.conv2d(x, filters=f, kernel_size=[3,3], \
                activation=tf.tanh, padding='same', trainable=True)
            conv = tf.layers.conv2d(conv, filters=f, kernel_size=[3,3], \
                activation=tf.tanh, padding='same', trainable=True)
            x    = tf.layers.max_pooling2d(conv, 3, 3, padding='same')

        return x

    def feed_forward(self, x):

        for n in range(par['n_layers']-1):

            W = self.var_dict['layer'+str(n)]['W']
            b = self.var_dict['layer'+str(n)]['b']

            if n < par['n_layers']-2:
                x = tf.nn.relu(x @ W + b)
            else:
                y = x @ W + b

        return y


    def optimize(self):

        meta_opt = tf.train.GradientDescentOptimizer(learning_rate=par['meta_learning_rate'])
        task_opt = tf.train.GradientDescentOptimizer(learning_rate=par['task_learning_rate'])
        eps = 1e-7

        ### Reptile algorithm
        self.backup_vars = [tf.get_variable(var.op.name+'_backup', initializer=var, trainable=False) for var in tf.trainable_variables()]

        k_opt_group = []
        for x, y_hat in zip(self.input_data, self.output_data):
            y = self.run_model(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_hat+eps))
            k_opt_group.append(meta_opt.minimize(loss))

        assigns = []
        with tf.control_dependencies(k_opt_group):
            for var, backup_var in zip(tf.trainable_variables(), self.backup_vars):
                assigns.append(tf.assign(var, backup_var + self.step_size*(var-backup_var)))
        self.pre_train = tf.group(*assigns)

        ### Regular training
        x = self.input_data[0]
        y_hat = self.output_data[0]
        y = self.run_model(x)
        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_hat+eps))
        self.train_op = task_opt.minimize(self.task_loss)

        correct_prediction = tf.equal(tf.argmax(y,-1), tf.argmax(y_hat,-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.stored_vars = [tf.get_variable(var.op.name+'_stored', initializer=var, trainable=False) for var in tf.trainable_variables()]
        self.make_stored_vars = self.store_vars()
        self.load_stored_vars = self.load_vars()


    def store_vars(self):
        assigns = []
        for var, stored_var in zip(tf.trainable_variables(), self.stored_vars):
            assigns.append(tf.assign(stored_var, var))
        return tf.group(*assigns)


    def load_vars(self):
        assigns = []
        for var, stored_var in zip(tf.trainable_variables(), self.stored_vars):
            assigns.append(tf.assign(var, stored_var))
        return tf.group(*assigns)



def main(save_fn='testing', gpu_id=None):

    print_parameters()

    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [par['k_steps'], None, *par['input_shape']], 'input')
    y = tf.placeholder(tf.float32, [par['k_steps'], None, par['n_output']], 'output')
    s = tf.placeholder(tf.float32, [], 'step')

    stim = stimulus.Stimulus()

    with tf.Session() as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, s)

        sess.run(tf.global_variables_initializer())

        print('\nPre-Training... [{} steps]'.format(par['k_steps']))
        for i in range(par['pre_train_batches']):

            meta_task = np.random.choice(par['n_meta_tasks'])
            step_size = par['epsilon']*(1-i/(par['pre_train_batches']))

            stims, ys = make_stim_batch(par['pre_train_batch_size'], meta_task, False, stim)

            _, loss, acc = sess.run([model.pre_train, model.task_loss, model.accuracy],\
                feed_dict={x:stims, y:ys, s:step_size})

            if i%500 == 0:
                print('Iter. {:5} | Loss: {:5.3f} | Acc: {:5.3f}'.format(i, loss, acc))

        sess.run(model.make_stored_vars)

        print('Training and testing... [{}-shot, {}-way]'.format(par['shot_batch_size'], par['n_test_tasks']))
        acc_list = []
        for i in range(par['testing_repetitions']):

            sess.run(model.load_stored_vars)

            train_stims, test_stims = [], []
            train_ys, test_ys = [], []
            for task in range(par['n_test_tasks']*i, par['n_test_tasks']*(i+1)):
                stims, ys = make_stim_batch(par['shot_batch_size']+1, task, True, stim)
                train_stims.append(stims[:,:-1,:])
                train_ys.append(ys[:,:-1,:])

                test_stims.append(stims[:,-1:,:])
                test_ys.append(ys[:,-1:,:])


            for j in range(par['eval_iterations']):
                mean_loss = []
                for task in range(par['n_test_tasks']):
                    stims = train_stims[task]
                    ys = train_ys[task]

                    _, loss = sess.run([model.train_op, model.task_loss], feed_dict={x:stims, y:ys})
                    mean_loss.append(loss)

                if j%(par['eval_iterations']-1) == 0:
                    print('Iter. {:3} | Mean Loss: {:5.3f}'.format(j, np.mean(mean_loss)))


            for task in range(par['n_test_tasks']):
                stims = test_stims[task]
                ys = test_ys[task]

                loss, acc = sess.run([model.task_loss, model.accuracy], feed_dict={x:stims, y:ys})
                acc_list.append(acc)
                print('Task {:2} | Loss: {:5.3f} | Acc: {:5.3f}'.format(task, loss, acc))
            print('Mean evaluation accuracy for k={} at rep {}: {:5.3f}%\n'.format(par['k_steps'], i, 100*np.mean(acc_list)))



def make_stim_batch(batch_size, task, test, stim):

    stims, ys = [], []
    for k in range(par['k_steps']):
        stim_in, y_hat, _ = stim.make_batch(batch_size, task, test=test)
        stims.append(stim_in)
        ys.append(y_hat)

    return np.stack(stims, axis=0), np.stack(ys, axis=0)


def print_parameters():
    print('{}-shot, {}-way, {} steps'.format(par['shot_batch_size'], par['n_test_tasks'], par['k_steps']))
    print('Batch size of {} for pre-training'.format(par['pre_train_batch_size']))


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
