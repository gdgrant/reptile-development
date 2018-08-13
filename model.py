### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product

# Model modules
from parameters import *
import cognitive_stimulus as stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

    """ RNN model for supervised and reinforcement learning training """

    def __init__(self, input_data, target_data, mask, gating):

        # Load input activity, target data, training mask, etc.
        self.input_data         = tf.unstack(input_data, axis=0)
        self.target_data        = tf.unstack(target_data, axis=0)
        self.gating             = tf.reshape(gating, [1,-1])
        self.time_mask          = tf.unstack(mask, axis=0)

        self.batch_by_one = tf.ones_like(self.input_data[0][:,0:1])

        # Declare all Tensorflow variables
        self.declare_variables()

        # Build the Tensorflow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()


    def declare_variables(self):
        """ Initialize all required variables """

        # All the possible prefixes based on network setup
        lstm_var_prefixes   = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
        bio_var_prefixes    = ['W_in', 'b_rnn', 'W_rnn']
        rl_var_prefixes     = ['W_pol_out', 'b_pol_out', 'W_val_out', 'b_val_out']
        base_var_prefies    = ['W_out', 'b_out']

        # Add relevant prefixes to variable declaration
        prefix_list = base_var_prefies
        if par['architecture'] == 'LSTM':
            prefix_list += lstm_var_prefixes
        elif par['architecture'] == 'BIO':
            prefix_list += bio_var_prefixes

        if par['training_method'] == 'RL':
            prefix_list += rl_var_prefixes
        elif par['training_method'] == 'SL':
            pass

        # Use prefix list to declare required variables and place them in a dict
        self.var_dict = {}
        with tf.variable_scope('network'):
            for p in prefix_list:
                self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])

        if par['architecture'] == 'BIO':
            # Modify recurrent weights if using EI neurons (in a BIO architecture)
            self.W_rnn_eff = (tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['W_rnn'])) \
                if par['EI'] else self.var_dict['W_rnn']

        with tf.variable_scope('init'):
            self.var_dict['h_init'] = tf.get_variable('h', initializer=0.1*tf.ones([1,par['n_hidden']]))
            if par['architecture'] == 'LSTM':
                self.var_dict['c_init'] = tf.get_variable('c', initializer=0.1*tf.ones([1,par['n_hidden']]))


    def rnn_cell_loop(self):
        """ Initialize parameters and execute loop through
            time to generate the network outputs """

        # Specify training method outputs
        self.output = []
        self.mask = [self.batch_by_one]
        if par['training_method'] == 'RL':
            self.pol_out = self.output  # For interchangeable use
            self.val_out = []
            self.action = []
            self.reward = [self.batch_by_one @ tf.constant(0., shape=[1,par['n_val']])]

        # Initialize state records
        self.h      = []
        #self.syn_x  = []
        #self.syn_u  = []

        # Initialize network state
        if par['architecture'] == 'BIO':
            h = self.gating * (self.batch_by_one @ self.var_dict['h_init'])
            c = tf.ones_like(h)
        elif par['architecture'] == 'LSTM':
            h = self.gating * (self.batch_by_one @ self.var_dict['h_init'])
            c = self.gating * (self.batch_by_one @ self.var_dict['c_init'])

        syn_x, syn_u = [], []
        for i in range(par['n_hidden']):
            t = self.batch_by_one
            if par['synapse_type'][i] == 1:
                sx = 1 * t
                su = 0.15 * t
            elif par['synapse_type'][i] == 2:
                sx = 1 * t
                su = 0.45 * t
            syn_x.append(sx)
            syn_u.append(su)

        syn_x = tf.stack(syn_x, axis=1)[...,0]
        syn_u = tf.stack(syn_u, axis=1)[...,0]
        mask  = self.mask[0]

        # Loop through the neural inputs, indexed in time
        for rnn_input, target, time_mask in zip(self.input_data, self.target_data, self.time_mask):

            # Compute the state of the hidden layer
            h, c, syn_x, syn_u = self.recurrent_cell(h, c, syn_x, syn_u, rnn_input)

            # Record hidden state
            self.h.append(h)
            #self.syn_x.append(syn_x)
            #self.syn_u.append(syn_u)

            if par['training_method'] == 'SL':
                # Compute outputs for loss
                y = h @ self.var_dict['W_out'] + self.var_dict['b_out']

                # Record supervised outputs
                self.output.append(y)

            elif par['training_method'] == 'RL':
                # Compute outputs for action
                pol_out        = h @ self.var_dict['W_pol_out'] + self.var_dict['b_pol_out']
                action_index   = tf.multinomial(pol_out, 1)
                action         = tf.one_hot(tf.squeeze(action_index), par['n_pol'])

                # Compute outputs for loss
                pol_out        = tf.nn.softmax(pol_out, axis=1)  # Note softmax for entropy loss
                val_out        = h @ self.var_dict['W_val_out'] + self.var_dict['b_val_out']

                # Check for trial continuation (ends if previous reward was non-zero)
                continue_trial = tf.cast(tf.equal(self.reward[-1], 0.), tf.float32)
                mask          *= continue_trial
                reward         = tf.reduce_sum(action*target, axis=1, keep_dims=True)*mask*tf.reshape(time_mask,[par['batch_size'], 1])

                # Record RL outputs
                self.pol_out.append(pol_out)
                self.val_out.append(val_out)
                self.action.append(action)
                self.reward.append(reward)

            # Record mask (outside if statement for cross-comptability)
            self.mask.append(mask)

        # Reward and mask trimming where necessary
        self.mask = self.mask[1:]
        if par['training_method'] == 'RL':
            self.reward = self.reward[1:]


    def recurrent_cell(self, h, c, syn_x, syn_u, rnn_input):
        """ Using the appropriate recurrent cell
            architecture, compute the hidden state """

        if par['architecture'] == 'BIO':

            # Apply synaptic short-term facilitation and depression, if required
            if par['synapse_config'] == 'std_stf':
                syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
                syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_u*syn_x*h
            else:
                h_post = h

            # Compute hidden state
            h = self.gating*tf.nn.relu((1-par['alpha_neuron'])*h \
              + par['alpha_neuron']*(rnn_input @ self.var_dict['W_in'] + h_post @ self.W_rnn_eff + self.var_dict['b_rnn']) \
              + tf.random_normal(tf.shape(h), 0, par['noise_rnn'], dtype=tf.float32))
            c = tf.constant(-1.)

        elif par['architecture'] == 'LSTM':

            # Compute LSTM state
            # f : forgetting gate, i : input gate,
            # c : cell state, o : output gate
            f   = tf.sigmoid(rnn_input @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
            i   = tf.sigmoid(rnn_input @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
            cn  = tf.tanh(rnn_input @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
            c   = f * c + i * cn
            o   = tf.sigmoid(rnn_input @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])

            # Compute hidden state
            h = self.gating * o * tf.tanh(c)
            syn_x = tf.constant(-1.)
            syn_u = tf.constant(-1.)

        return h, c, syn_x, syn_u


    def optimize(self):
        """ Calculate losses and apply corrections to model """

        # Set up optimizer and required constants
        epsilon = 1e-7
        adam_optimizer = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

        # Spiking activity loss
        self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
            for (h, mask, time_mask) in zip(self.h, self.mask, self.time_mask)]))

        # Training-specific losses
        if par['training_method'] == 'SL':
            RL_loss = tf.constant(0.)

            # Task loss (cross entropy)
            self.pol_loss = tf.reduce_mean([mask*tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, \
                labels=target, dim=1) for y, target, mask in zip(self.output, self.target_data, self.time_mask)])
            sup_loss = self.pol_loss

        elif par['training_method'] == 'RL':
            sup_loss = tf.constant(0.)

            # Collect information from across time
            self.time_mask  = tf.reshape(tf.stack(self.time_mask),(par['num_time_steps'], par['batch_size'], 1))
            self.mask       = tf.stack(self.mask)
            self.reward     = tf.stack(self.reward)
            self.action     = tf.stack(self.action)
            self.pol_out    = tf.stack(self.pol_out)

            # Compute predicted value, the actual action taken, and the advantage for plugging into the policy loss
            val_out = tf.stack(self.val_out)
            val_out_stacked = tf.concat([val_out, tf.zeros([1,par['batch_size'],par['n_val']])], axis=0)
            terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward, tf.constant(0.))), tf.float32)
            pred_val = self.reward + par['discount_rate']*val_out_stacked[1:,:,:]*(1-terminal_state)
            advantage = pred_val - val_out_stacked[:-1,:,:]

            # Stop gradients back through action, advantage, and mask
            action_static = tf.stop_gradient(self.action)
            advantage_static = tf.stop_gradient(advantage)
            mask_static = tf.stop_gradient(self.mask)

            # Policy loss
            self.pol_loss = -tf.reduce_mean(advantage_static*mask_static*self.time_mask*action_static*tf.log(epsilon+self.pol_out))

            # Value loss
            self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(mask_static*self.time_mask*tf.square(val_out_stacked[:-1,:,:]-pred_val))

            # Entropy loss
            self.entropy_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(mask_static*self.time_mask*self.pol_out*tf.log(epsilon+self.pol_out), axis=1))

            # Collect RL losses
            RL_loss = self.pol_loss + self.val_loss - self.entropy_loss

        # Collect loss terms and compute gradients
        total_loss = sup_loss + RL_loss + self.spike_loss
        self.train_op = adam_optimizer.compute_gradients(total_loss)

        # Make reset operations
        self.reset_adam_op = adam_optimizer.reset_params()

        # Make saturation correction operation
        self.make_recurrent_weights_positive()


    def make_recurrent_weights_positive(self):
        """ Very slightly de-saturate recurrent weights """

        reset_weights = []
        for var in tf.trainable_variables():
            if 'W_rnn' in var.op.name:
                # make all negative weights slightly positive
                reset_weights.append(tf.assign(var,tf.maximum(1e-9, var)))

        self.reset_rnn_weights = tf.group(*reset_weights)


def supervised_learning(save_fn='test.pkl', gpu_id=None):
    """ Run supervised learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x = tf.placeholder(tf.float32, [par['num_time_steps'], None, par['n_input']], 'stim')
    y = tf.placeholder(tf.float32, [par['num_time_steps'], None, par['n_output']], 'out')
    m = tf.placeholder(tf.float32, [par['num_time_steps'], None], 'mask')
    g = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')

    # Set up stimulus and accuracy recording
    stim = stimulus.MultiStimulus()
    accuracy_full = []
    accuracy_grid = np.zeros([par['n_tasks'],par['n_tasks']])
    full_activity_list = []

    # Display relevant parameters
    print_key_info()

    # Start Tensorflow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, m, g)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        # Begin training loop, iterating over tasks
        for task in range(par['n_tasks']):
            for i in range(par['pre_train_batches']):

                # Generate a batch of stimulus data for training
                name, stim_in, y_hat, mk, _ = stim.generate_trial(task, par['pre_train_batch_size'])

                # Put together the feed dictionary
                feed_dict = {x:stim_in, y:y_hat, g:par['gating'][task], m:mk}

                # Run the model using one of the available stabilization methods
                _, loss, spike_loss, output = sess.run([model.train_op, model.pol_loss, \
                    model.spike_loss, model.output], feed_dict=feed_dict)

                # Display network performance
                if i%25 == 0:
                    acc = get_perf(y_hat, output, mk, par['pre_train_batch_size'])
                    print('Iter {} | Task name {} | Accuracy {} | Loss {} | Spike Loss {}'.format(\
                        i, name, acc, loss, spike_loss))

            quit('Quit at end of task 0.')
            # Test all tasks at the end of each learning session
            num_reps = 10
            task_activity_list = []
            for task_prime in range(task+1):
                for r in range(num_reps):

                    # Generate stimulus batch for testing
                    name, stim_in, y_hat, mk, _ = stim.generate_trial(task_prime, par['batch_size'])

                    # Assemble feed dict and run model
                    feed_dict = {x:stim_in, g:par['gating'][task_prime]}
                    output, h = sess.run([model.output, model.h], feed_dict=feed_dict)

                    # Record results
                    acc = get_perf(y_hat, output, mk)
                    accuracy_grid[task,task_prime] += acc/num_reps

                # Record network activity
                task_activity_list.append(h)

            # Aggregate task after testing each task set
            # Each of [all tasks] elements is [tasks tested, time steps, batch size hidden size]
            full_activity_list.append(task_activity_list)

            # Display accuracy grid after testing is complete
            print('Accuracy grid after task {}:'.format(task))
            print(accuracy_grid[task,:])
            print()

            # Reset the Adam Optimizer and save previous parameter values as current ones
            sess.run(model.reset_adam_op)

            # Reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(model.reset_weights)

        if par['save_analysis']:
            save_results = {'task': task, 'accuracy_grid': accuracy_grid, 'par': par, 'activity': full_activity_list}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete. (Supervised)')


def reinforcement_learning(save_fn='test.pkl', gpu_id=None):
    """ Run reinforcement learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x, target, mask, pred_val, actual_action, \
        advantage, mask, gating = generate_placeholders()

    # Set up stimulus and accuracy recording
    stim = stimulus.MultiStimulus()
    accuracy_full = []
    accuracy_grid = np.zeros([par['n_tasks'],par['n_tasks']])
    full_activity_list = []
    model_performance = {'reward': [], 'entropy_loss': [], 'val_loss': [], 'pol_loss': [], 'spike_loss': [], 'trial': [], 'task': []}
    reward_matrix = np.zeros((par['n_tasks'], par['n_tasks']))

    # Display relevant parameters
    print_key_info()

    # Start Tensorflow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            # Check order against args unpacking in model if editing
            model = Model(x, target, mask, gating)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        # Begin training loop, iterating over tasks
        for task in range(par['n_tasks']):
            accuracy_iter = []
            task_start_time = time.time()

            for i in range(par['n_train_batches']):

                # Generate a batch of stimulus data for training
                name, input_data, _, mk, reward_data = stim.generate_trial(task, par['batch_size'])
                mk = mk[...,np.newaxis]

                # Put together the feed dictionary
                feed_dict = {x:input_data, target:reward_data, mask:mk, gating:par['gating'][task]}

                _, _, pol_loss, val_loss, spike_loss, ent_loss, h_list, reward = \
                    sess.run([model.train_op, model.update_current_reward, model.pol_loss, \
                    model.val_loss, model.spike_loss, model.entropy_loss, model.h, \
                    model.reward], feed_dict=feed_dict)

                # Record accuracies
                reward = np.stack(reward_list)
                acc = np.mean(np.sum(reward>0,axis=0))
                accuracy_iter.append(acc)
                if i > 2000:
                    if np.mean(accuracy_iter[-2000:]) > 0.985 or (i>25000 and np.mean(accuracy_iter[-2000:]) > 0.98):
                        print('Accuracy reached threshold')
                        break

                # Display network performance
                if i%500 == 0:
                    print('Iter ', i, 'Task name ', name, ' accuracy', acc, ' aux loss', aux_loss, \
                    'mean h', np.mean(np.stack(h_list)), 'time ', np.around(time.time() - task_start_time))

            # Test all tasks at the end of each learning session
            num_reps = 10
            task_activity_list = []
            for task_prime in range(task+1):
                for r in range(num_reps):

                    # make batch of training data
                    name, input_data, _, mk, reward_data = stim.generate_trial(task_prime, par['batch_size'])
                    mk = mk[..., np.newaxis]

                    reward_list, h = sess.run([model.reward, model.h], feed_dict = {x:input_data, target: reward_data, \
                        gating:par['gating'][task_prime], mask:mk})
                    # TODO: figure out what's with the extra dimension at index 0 in reward
                    reward = np.squeeze(np.stack(reward_list))
                    reward_matrix[task,task_prime] += np.mean(np.sum(reward>0,axis=0))/num_reps

                # Record network activity
                task_activity_list.append(h)

            # Aggregate task after testing each task set
            # Each of [all tasks] elements is [tasks tested, time steps, batch size hidden size]
            full_activity_list.append(task_activity_list)

            print('Accuracy grid after task {}:'.format(task))
            print(reward_matrix[task,:])

            results = {'reward_matrix': reward_matrix, 'par': par, 'activity': full_activity_list}
            pickle.dump(results, open(par['save_dir'] + save_fn, 'wb') )
            print('Analysis results saved in', save_fn)
            print('')

            # Reset the Adam Optimizer, and set the previous parameter values to their current values
            sess.run(model.reset_adam_op)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

    print('\nModel execution complete. (Reinforcement)')


def print_key_info():
    """ Display requested information """

    if par['training_method'] == 'SL':
        key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost',\
            'n_hidden','noise_rnn_sd','learning_rate','gating_type', 'gate_pct']
    elif par['training_method'] == 'RL':
        key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost',\
            'n_hidden','noise_rnn_sd','learning_rate','discount_rate', 'mask_duration',\
            'gating_type', 'gate_pct','fix_break_penalty','wrong_choice_penalty',\
            'correct_choice_reward','include_rule_signal']
    print('Key info:')
    print('-'*40)
    for k in key_info:
        print(k, ' ', par[k])
    print('-'*40)


def print_reinforcement_results(iter_num, model_performance):
    """ Aggregate and display reinforcement learning results """

    reward = np.mean(np.stack(model_performance['reward'])[-par['iters_between_outputs']:])
    pol_loss = np.mean(np.stack(model_performance['pol_loss'])[-par['iters_between_outputs']:])
    val_loss = np.mean(np.stack(model_performance['val_loss'])[-par['iters_between_outputs']:])
    entropy_loss = np.mean(np.stack(model_performance['entropy_loss'])[-par['iters_between_outputs']:])

    print('Iter. {:4d}'.format(iter_num) + ' | Reward {:0.4f}'.format(reward) +
      ' | Pol loss {:0.4f}'.format(pol_loss) + ' | Val loss {:0.4f}'.format(val_loss) +
      ' | Entropy loss {:0.4f}'.format(entropy_loss))


def get_perf(target, output, mask, batch_size):
    """ Calculate task accuracy by comparing the actual network output
    to the desired output only examine time points when test stimulus is
    on in another words, when target[:,:,-1] is not 0 """

    output = np.stack(output, axis=0)
    mk = mask*np.reshape(target[:,:,-1] == 0, (batch_size, par['num_time_steps'], 1))

    target = np.argmax(target, axis = 2)
    output = np.argmax(output, axis = 2)

    return np.sum(np.float32(target == output)*np.squeeze(mk))/np.sum(mk)


def append_model_performance(model_performance, reward, entropy_loss, pol_loss, val_loss, trial_num):

    reward = np.mean(np.sum(reward,axis = 0))/par['trials_per_sequence']
    model_performance['reward'].append(reward)
    model_performance['entropy_loss'].append(entropy_loss)
    model_performance['pol_loss'].append(pol_loss)
    model_performance['val_loss'].append(val_loss)
    model_performance['trial'].append(trial_num)

    return model_performance


def generate_placeholders():

    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], None, 1])
    x = tf.placeholder(tf.float32, shape=[par['num_time_steps'], None, par['n_input']])  # input data
    target = tf.placeholder(tf.float32, shape=[par['num_time_steps'], None, par['n_pol']])  # input data
    pred_val = tf.placeholder(tf.float32, shape=[par['num_time_steps'], None, par['n_val'], ])
    actual_action = tf.placeholder(tf.float32, shape=[par['num_time_steps'], None, par['n_pol']])
    advantage  = tf.placeholder(tf.float32, shape=[par['num_time_steps'], None, par['n_val']])
    gating = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')

    return x, target, mask, pred_val, actual_action, advantage, mask, gating


def main(save_fn='testing', gpu_id=None):

    # Update all dependencies in parameters
    update_dependencies()

    # Identify learning method and run accordingly
    if par['training_method'] == 'SL':
        supervised_learning(save_fn, gpu_id)
    elif par['training_method'] == 'RL':
        reinforcement_learning(save_fn, gpu_id)
    else:
        raise Exception('Select a valid learning method.')


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main('testing.pkl', sys.argv[1])
        else:
            main('testing.pkl')
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')
