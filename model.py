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

        x = tf.reshape(x, [-1, np.product(x.shape[1:])])
        y = self.feed_forward(x)

        return y


    def convolutional(self, x):

        for i in range(4):
            conv = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, \
                padding='same', trainable=True, name='conv{}'.format(i))
            norm = tf.layers.batch_normalization(conv, training=True)
            x = tf.nn.relu(norm)

        return x


    def feed_forward(self, x):

        for n in range(par['n_layers']-1):

            W = self.var_dict['layer'+str(n)]['W']
            b = self.var_dict['layer'+str(n)]['b']

            if n < par['n_layers']-2:
                x = tf.nn.relu(x @ W) + b
                x = tf.layers.dropout(x, rate=par['dropout_drop_pct'])
            else:
                y = x @ W + b

        return y


    def optimize(self):

        #meta_opt = tf.train.GradientDescentOptimizer(learning_rate=par['meta_learning_rate'])
        #task_opt = tf.train.GradientDescentOptimizer(learning_rate=par['task_learning_rate'])
        task_opt = tf.train.AdamOptimizer(learning_rate=par['task_learning_rate'], beta1=0.)
        eps = 1e-7

        x = self.input_data[0]
        y_hat = self.output_data[0]
        y = self.run_model(x)


        ### Reptile algorithm
        self.backup_vars = [tf.get_variable(var.op.name+'_backup', initializer=var, trainable=False) for var in tf.trainable_variables()]
        self.all_vars = {var.op.name : var for var in tf.trainable_variables()}

        self.make_saved_inter_vars = self.save_inter_vars()
        self.interpolate_saved_inter_vars = self.interpolate_from_saved_vars()

        """k_opt_group = [tf.no_op()]
        for k, (x, y_hat) in enumerate(zip(self.input_data, self.output_data)):
            #with tf.control_dependencies([k_opt_group[-1]]):
            y = self.run_model(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_hat))
            meta_opt.minimize(loss)
                #k_opt_group.append(meta_opt.minimize(loss))

        assigns = []
        with tf.control_dependencies(k_opt_group):
            for var, backup_var in zip(tf.trainable_variables(), self.backup_vars):
                assigns.append(tf.assign(var, backup_var + self.step_size*(var-backup_var)))
        self.pre_train = tf.group(*assigns)"""


        ### Regular training
        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_hat))
        self.train_op = task_opt.minimize(self.task_loss)
        self.pre_train = self.train_op

        correct_prediction = tf.equal(tf.argmax(y,-1), tf.argmax(y_hat,-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.stored_vars = [tf.get_variable(var.op.name+'_stored', initializer=var, trainable=False) for var in tf.trainable_variables()]
        self.make_stored_vars = self.store_vars()
        self.load_stored_vars = self.load_vars()


    def save_inter_vars(self):
        assigns = []
        for var, stored_var in zip(tf.trainable_variables(), self.backup_vars):
            assigns.append(tf.assign(stored_var, var))
        return tf.group(*assigns)


    def interpolate_from_saved_vars(self):
        assigns = []
        for var, stored_var in zip(tf.trainable_variables(), self.backup_vars):
            assigns.append(tf.assign(var, stored_var + self.step_size*(var-stored_var)))
        return tf.group(*assigns)


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

    x = tf.placeholder(tf.float32, [par['k_steps_dummy'], None, *par['input_shape']], 'input')
    y = tf.placeholder(tf.float32, [par['k_steps_dummy'], None, par['n_output']], 'output')
    s = tf.placeholder(tf.float32, [], 'step')

    stim = stimulus.Stimulus()

    with tf.Session() as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, s)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        if not par['load_from_checkpoint']:

            print('\nPre-Training... [{} steps]'.format(par['k_steps']))
            for i in range(par['pre_train_batches']):

                stim.make_task(par['n_ways'])
                step_size = par['epsilon']*(1-i/(par['pre_train_batches']))

                sess.run(model.make_stored_vars)
                sess.run(model.make_saved_inter_vars)

                var_states = {name : 0. for name in sess.run(model.all_vars).keys()}
                for m in range(par['meta_batch_size']):
                    sess.run(model.load_stored_vars)

                    for k in range(par['k_steps']):
                        stims, ys = get_batch(stim, par['pre_train_batch_size'], test=False)
                        _, loss, acc = sess.run([model.pre_train, model.task_loss, model.accuracy],\
                            feed_dict={x:stims, y:ys})

                    new_vars = sess.run(model.all_vars)
                    for key in new_vars.keys():
                        var_states[key] += new_vars[key]/par['meta_batch_size']

                with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                    assigns = [tf.assign(tf.get_variable(key), var_states[key]) for key in var_states.keys()]
                sess.run(assigns)
                sess.run(model.interpolate_saved_inter_vars, feed_dict={s:step_size})

                if i%100 == 0:
                    print('Iter. {:5} | Loss: {:5.3f} | Acc: {:5.3f}'.format(i, loss, acc))

            save_path = saver.save(sess, './pretrained.ckpt')


        print('\nTraining and testing... [{}-shot, {}-way]'.format(par['n_shots'], par['n_ways']))
        acc_list = []
        for i in range(par['test_repetitions']):

            #sess.run(model.load_stored_vars)
            saver.restore(sess, './pretrained.ckpt')
            stim.make_task(3*par['n_shots'], eval=True)


            for j in range(par['eval_iterations']):
                mean_loss = []

                stims, ys = get_batch(stim, par['eval_batch_size'], test=False)

                _, loss = sess.run([model.train_op, model.task_loss], feed_dict={x:stims, y:ys})
                mean_loss.append(loss)

                if j%(par['eval_iterations']-1) == 0:
                    print('Iter. {:3} | Mean Loss: {:5.3f}'.format(j, np.mean(mean_loss)))


            for t in range(par['test_iterations']):

                stims, ys = get_batch(stim, par['test_batch_size'], test=True)

                loss, acc = sess.run([model.task_loss, model.accuracy], feed_dict={x:stims, y:ys})
                acc_list.append(acc)

            print('Mean evaluation accuracy for k={} at rep {}: {:5.3f}%\n'.format(par['k_steps'], i, 100*np.mean(acc_list)))


def get_batch(stim, batch_size, test):

    stims, ys = [], []
    for k in range(par['k_steps_dummy']):
        stim_in, y_hat, _ = stim.make_batch(batch_size, test=test)
        stims.append(stim_in)
        ys.append(y_hat)
    stims = np.stack(stims, axis=0)
    ys = np.stack(ys, axis=0)

    return stims, ys


def print_parameters():
    print('{}-shot, {}-way, {} steps'.format(par['n_shots'], par['n_ways'], par['k_steps']))
    print('Batch size of {} for pre-training'.format(par['pre_train_batch_size']))


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
