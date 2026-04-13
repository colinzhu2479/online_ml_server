from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

keras.backend.set_floatx('float64')
from tensorflow.keras import backend as K

def act_gaussian(std=5):
    return lambda x:K.exp((-x**2)/std)


def individual_normalize(x):
    xmin = np.min(x,axis=0)
    xrange =np.max(x,axis=0)-np.min(x,axis=0)
    x = np.array([(x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i])) for i in range(len(x[0]))]).T
    return x, xmin, xrange


def global_normalize(x,shift=0,x_has_min=True ):

    xrange = np.max(x)-np.min(x)
    if x_has_min:
        xmin=np.min(x)-shift*xrange
    else:
        xmin=0
    if xrange==0:
        print('Error: All coordinates are the same.')
        exit()
    x = np.array((x-xmin)/xrange)
    return x, xmin, xrange


def build_model(x_0, target_type, num_atom, num_node_ratio, num_layer, activation=act_gaussian()):
    model_e = Sequential()
    ## other available activation: "relu", "sigmoid", "softmax", "tanh" ......

    num_feature =  len(x_0)
    num_nodes = num_node_ratio * num_feature
    model_e.add(Dense(num_nodes, input_dim=num_feature, activation=activation, kernel_initializer='he_uniform'))

    for ii in range(num_layer-1):
        model_e.add(Dense(num_nodes, activation=activation, kernel_initializer='he_uniform'))

    if target_type == 'energy':
        model_e.add(Dense(1, kernel_initializer='he_uniform'))
    elif target_type == "force":
        model_e.add(Dense(num_atom, kernel_initializer='he_uniform'))
    else:
        print("Error: unknown target quantity.")
        exit()

    model_e.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model_e


def def_normalization(x, y, target_type, xshift=0, x_has_min=False):
    if target_type == 'force':

        x_t,x_min,x_range = individual_normalize(x)

        y_range = 1/600
        y_min = 0.0

        e_range = 1/600
        e_min = 0.0
    elif target_type == 'energy':
        x_t, x_min, x_range = global_normalize(x)

        y_t, y_min, y_range = global_normalize(y)

        e_range = 0.001
        e_min = -0.004

    else:
        print("Error: unknown target quantity.")
        exit()

    return x_range, x_min, y_range, y_min, e_range, e_min
