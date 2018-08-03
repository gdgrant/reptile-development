# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product
import matplotlib.pyplot as plt
plt.switch_backend('agg')       # For remote plot generation

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

    def __init__(self, input_data_batches, output_data_batches):

        self.input_data = tf.unstack(input_data_batches, axis=0)
        self.output_data = tf.unstack(output_data_batches, axis=0)

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

        for n in range(par['n_layers']-1):

            W = self.var_dict['layer'+str(n)]['W']
            b = self.var_dict['layer'+str(n)]['b']

            if n < par['n_layers']-2:
                x = tf.nn.relu(x @ W + b)
            else:
                y = x @ W + b

        return y


    def optimize(self):

        opt = tf.train.GradientDescentOptimizer(learning_rate=par['learning_rate'])
        eps = 1e-7

        ### Reptile algorithm
        self.backup_vars = [tf.get_variable(var.op.name+'_backup', initializer=var, trainable=False) for var in tf.trainable_variables()]

        k_opt_group = []
        for x, y_hat in zip(self.input_data, self.output_data):
            y = self.run_model(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_hat+eps))
            k_opt_group.append(opt.minimize(loss))

        assigns = []
        with tf.control_dependencies(k_opt_group):
            for var, backup_var in zip(tf.trainable_variables(), self.backup_vars):
                assigns.append(tf.assign(var, backup_var + par['epsilon']*(var-backup_var)))
        self.pre_train = tf.group(*assigns)

        ### Regular training
        x = self.input_data[0]
        y_hat = self.output_data[0]
        y = self.run_model(x)
        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_hat+eps))
        self.train_op = opt.minimize(self.task_loss)

        correct_prediction = tf.equal(tf.argmax(y,-1), tf.argmax(y_hat,-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main(save_fn='testing', gpu_id=None):

    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [par['k_steps'], None, par['n_input']], 'input')
    y = tf.placeholder(tf.float32, [par['k_steps'], None, par['n_output']], 'output')

    stim = stimulus.Stimulus()

    with tf.Session() as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y)

        sess.run(tf.global_variables_initializer())

        print('\nPre-Training... [{} steps]'.format(par['k_steps']))
        for i in range(par['n_meta_batches']*par['n_meta_tasks']):

            meta_task = np.random.choice(par['n_meta_tasks'])

            stims, ys, mks = [], [], []
            for k in range(par['k_steps']):
                stim_in, y_hat, mk = stim.make_batch(par['pre_train_batch_size'], meta_task, test=False)
                stims.append(stim_in)
                ys.append(y_hat)
                mks.append(mk)

            _, loss, acc = sess.run([model.pre_train, model.task_loss, model.accuracy],\
                feed_dict={x:np.stack(stims, axis=0), y:np.stack(ys, axis=0)})

            if i%100 == 0:
                print('Iter. {:4} | Loss: {:5.3f} | Acc: {:5.3f}'.format(i, loss, acc))


        print('\nTraining shots... [{}-shot, {}-way]'.format(par['shot_batch_size'], par['n_test_tasks']))
        for t in range(par['n_test_tasks']):
            task = par['n_meta_tasks'] + t

            stims, ys, mks = [], [], []
            for k in range(par['k_steps']):
                stim_in, y_hat, mk = stim.make_batch(par['shot_batch_size'], task, test=False)
                stims.append(stim_in)
                ys.append(y_hat)
                mks.append(mk)

            _, loss = sess.run([model.train_op, model.task_loss],\
                feed_dict={x:np.stack(stims, axis=0), y:np.stack(ys, axis=0)})
            print('Task {:2} | Loss: {:5.3f}'.format(task, loss))


        print('\nTesting...')
        for t in range(par['n_test_tasks']):
            task = par['n_meta_tasks'] + t

            stims, ys, mks = [], [], []
            for k in range(par['k_steps']):
                stim_in, y_hat, mk = stim.make_batch(par['test_batch_size'], task, test=True)
                stims.append(stim_in)
                ys.append(y_hat)
                mks.append(mk)

            loss, acc = sess.run([model.task_loss, model.accuracy],\
                feed_dict={x:np.stack(stims, axis=0), y:np.stack(ys, axis=0)})
            print('Task {:2} | Loss: {:5.3f} | Acc: {:5.3f}'.format(task, loss, acc))




if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')