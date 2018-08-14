import model
from parameters import *
import pickle

def run_model():

    try:
        accuracy = model.main('testing.pkl', None)
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')
        quit()

    return accuracy



k5_params = {'k_steps' : 5, 'pre_train_batches' : int(1e3)}
k1_params = {'k_steps' : 1, 'pre_train_batches' : int(5e3)}

for j in range(10000):

    update_parameters(k5_params)
    acc = run_model()
    print('k=5:')
    print(acc)
    print('\n\n')
    #pickle.dump(acc, open('./savedir/k5_v{}.pkl'.format(j), 'wb'))

    update_parameters(k1_params)
    acc = run_model()
    print('k=1:')
    print(acc)
    print('\n\n')
    #pickle.dump(acc, open('./savedir/k1_v{}.pkl'.format(j), 'wb'))
