### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
from itertools import product

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # General parameters
    'save_dir'              : './savedir/',
    'meta_learning_rate'    : 0.001,
    'task_learning_rate'    : 0.001,
    'task'                  : 'omniglot',
    'conv_input'            : True,

    # Task specs
    'n_tasks'               : 100,          # Randomly pre-generated MNIST tasks
    'hidden_layers'         : [400, 400],

    # Reptile-specific parameters
    'n_meta_tasks'          : 20,
    'n_test_tasks'          : 5,
    'k_steps'               : 5,
    'epsilon'               : 1.,

    # Dropout
    'dropout_drop_pct'      : 0.0,
    'input_drop_keep_pct'   : 1.0,
    'conv_drop_keep_pct'    : 0.75,

    # Training specs
    'test_batch_size'       : 200,
    'pre_train_batch_size'  : 10,
    'pre_train_batches'     : int(1e5),
    'shot_batch_size'       : 5,
    'eval_iterations'       : 50,
    'testing_repetitions'   : 100,

}

############################
### Dependent parameters ###
############################


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    if par['task'] == 'mnist':
        par['input_shape'] = [28, 28]
        par['n_input'] = np.product(par['input_shape'])
        par['n_output'] = 10
    elif par['task'] == 'omniglot':
        par['input_shape'] = [105, 105]
        par['n_input'] = 3136 if par['conv_input'] else np.product(par['input_shape'])
        par['n_output'] = par['n_meta_tasks'] + par['n_test_tasks']

    par['layer_dims'] = [par['n_input']] + par['hidden_layers'] + [par['n_output']]


    par['n_layers'] = len(par['layer_dims'])
    if par['task'] == 'mnist' or par['task'] == 'imagenet':
        par['labels_per_task'] = 10
    elif par['task'] == 'cifar':
        par['labels_per_task'] = 5


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating:', key, '-->', val)
    update_dependencies()


update_dependencies()
print("--> Parameters successfully loaded.\n")
