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
    'learning_rate'         : 0.001,
    'task'                  : 'mnist',

    # Task specs
    'n_tasks'               : 100,
    'layer_dims'            : [28**2, 400, 400, 10],

    # Reptile-specific parameters
    'n_meta_tasks'          : 20,
    'n_test_tasks'          : 5,
    'k_steps'               : 5,
    'epsilon'               : 1.,

    # Dropout
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'conv_drop_keep_pct'    : 0.75,

    # Training specs
    'test_batch_size'       : 200,
    'pre_train_batch_size'  : 200,
    'shot_batch_size'       : 5,
    'n_meta_batches'        : 201,
    'n_test_batches'        : 201,

}

############################
### Dependent parameters ###
############################


def update_dependencies():
    """
    Updates all parameter dependencies
    """
    par['n_input'] = par['layer_dims'][0]
    par['n_output'] = par['layer_dims'][-1]

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
