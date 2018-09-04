from parameters import *
import model
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')       # For remote plot generation


k1_params = {'k_steps' : 1, 'pre_train_batches' : int(5e4)}
k5_params = {'k_steps' : 5, 'pre_train_batches' : int(1e4)}
load_params = {'load_from_checkpoint' : True}
eval_params = {'eval_iterations' : 50, 'eval_batch_size' : 5}


def run_model():
    try:
        acc = model.main(gpu_id='3')
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')

    return acc


def pretrain_iters_sweep():

    for iters in [1, 10, 100, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]:
        for k in [1, 5]:
            params = {'k_steps' : k, 'pre_train_batches' : iters*(6-k)}
            update_parameters(params)
            update_dependencies()
            acc = run_model()
            print('Acc for k={} at {} iterations: {} \n\n\n'.format(k, iters*(6-k), acc))
            pickle.dump(acc, open('./savedir/omniglot_sweep_k{}_by_iter{}.pkl'.format(k, iters), 'wb'))


def repetitions():

    for j in range(10000):

        update_parameters(k5_params)
        update_dependencies()
        acc = run_model()
        print('Acc for k=5:', acc, '\n\n')
        pickle.dump(acc, open('./savedir/omniglot_j1_k5_v{}.pkl'.format(j), 'wb'))

        update_parameters(k1_params)
        update_dependencies()
        acc = run_model()
        print('Acc for k=1:', acc, '\n\n')
        pickle.dump(acc, open('./savedir/omniglot_j1_k1_v{}.pkl'.format(j), 'wb'))


def sweep():

    update_parameters(load_params)

    setup = list(range(1000,2001,50))[1:]

    acc_array = np.zeros([2, len(setup), 3])

    for i in setup:

        print('-'*80)
        update_parameters({'eval_iterations' : i, 'eval_batch_size' : 5})

        for j in range(3):

            update_parameters(k5_params)
            update_dependencies()
            acc_array[1, (i-1001)//50, j] = run_model()

            update_parameters(k1_params)
            update_dependencies()
            acc_array[0, (i-1001)//50, j] = run_model()

    acc_array = np.mean(acc_array, axis=-1)

    plt.scatter(setup, acc_array[0], c='r', label='k=1')
    plt.scatter(setup, acc_array[1], c='b', label='k=5')
    plt.legend()
    plt.title('Number of Iterations vs. Accuracy')
    plt.xlabel('Num. Eval. Iterations')
    plt.ylabel('Testing Accuracy')
    plt.savefig('iters_sweep_grand.png')






pretrain_iters_sweep()
