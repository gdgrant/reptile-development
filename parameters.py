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
    'load_from_checkpoint'  : False,
    'meta_learning_rate'    : 0.001,
    'task_learning_rate'    : 0.001,
    'task'                  : 'omniglot',
    'conv_input'            : True,
    'dropout_drop_pct'      : 0.0,
    'hidden_layers'         : [],

    # Reptile-specific parameters
    'k_steps'               : 5,
    'k_steps_dummy'         : 1,
    'n_ways'                : 5,
    'n_shots'               : 5,
    'epsilon'               : 1.,

    # Training specs
    'meta_batch_size'       : 5,

    'pre_train_batch_size'  : 10,
    'pre_train_batches'     : int(1e4),

    'eval_iterations'       : 50,
    'eval_batch_size'       : 5,

    'test_repetitions'      : 1000,
    'test_iterations'       : 20,
    'test_batch_size'       : 100,

}

############################
### Dependent parameters ###
############################


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    if par['task'] == 'mnist':
        par['n_tasks'] = 100
        par['input_shape'] = [28, 28]
        par['n_input'] = np.product(par['input_shape'])
        par['n_output'] = 10
    elif par['task'] == 'omniglot':
        par['input_shape'] = [26, 26]
        par['n_input'] = 256 if par['conv_input'] else np.product(par['input_shape'])
        par['n_output'] = par['n_ways'] #par['n_meta_tasks'] + par['n_test_tasks']

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
