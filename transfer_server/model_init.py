"""
Model Initialization Module for Molecular Potential Energy Surfaces.

This module is part of a machine learning pipeline for training and deploying 
neural network models of molecular potential energy surfaces and atomic forces. 
It is designed to support transfer learning, allowing models to adapt from small 
reference systems to larger target systems. 

By integrating graph representation (graph theoretic fragmentation) of molecules, 
this pipeline significantly reduces the computational cost of training samples, 
lowers neural network model complexity, and reduces required training epochs.
"""

from typing import Callable, Tuple, Any, Union
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Set keras backend float type for high-precision molecular computations
keras.backend.set_floatx('float64')
from tensorflow.keras import backend as K


def act_gaussian(std: float = 5) -> Callable:
    """
    Creates a Gaussian activation function for neural network layers.
    
    Args:
        std (float): The standard deviation of the Gaussian function (default: 5).
        
    Returns:
        Callable: A keras backend function representing the Gaussian activation.
    """
    return lambda x: K.exp((-x**2)/std)


def individual_normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes a dataset feature-by-feature (column-wise).
    
    Args:
        x (np.ndarray): The input data matrix.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Normalized data array.
            - Array of minimum values per feature.
            - Array of ranges (max - min) per feature.
    """
    xmin = np.min(x, axis=0)
    xrange = np.max(x, axis=0) - np.min(x, axis=0)
    # Normalize each column independently
    x_norm = np.array([(x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i])) for i in range(len(x[0]))]).T
    return x_norm, xmin, xrange


def global_normalize(x: np.ndarray, shift: float = 0, x_has_min: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    Normalizes a dataset globally across all features using the overall min and max.
    
    Args:
        x (np.ndarray): The input data matrix.
        shift (float): An optional shift factor to adjust the minimum (default: 0).
        x_has_min (bool): If True, computes minimum from data; if False, assumes min is 0 (default: True).
        
    Returns:
        Tuple[np.ndarray, float, float]: Normalized array, the scalar global min, and the global range.
    """
    xrange = np.max(x) - np.min(x)
    
    if x_has_min:
        xmin = np.min(x) - shift * xrange
    else:
        xmin = 0
        
    if xrange == 0:
        print('Error: All coordinates are the same.')
        exit()
        
    x_norm = np.array((x - xmin) / xrange)
    return x_norm, xmin, xrange


def build_model(
    x_0: np.ndarray, target_type: str, num_atom: int, num_node_ratio: int, 
    num_layer: int, activation: Callable = act_gaussian()
) -> Sequential:
    """
    Builds and compiles a Keras Sequential neural network model.
    
    Args:
        x_0 (np.ndarray): Sample input data to determine feature dimension.
        target_type (str): Type of prediction target ('energy' or 'force').
        num_atom (int): Number of atoms, used to determine the output dimension for 'force' models.
        num_node_ratio (int): Multiplier to determine the number of nodes per hidden layer relative to input features.
        num_layer (int): Total number of layers (including the input layer).
        activation (Callable): The activation function to use for hidden layers (default: act_gaussian()).
        
    Returns:
        Sequential: The compiled Keras model.
    """
    model_e = Sequential()
    
    num_feature = len(x_0)
    num_nodes = num_node_ratio * num_feature
    
    # Input layer
    model_e.add(Dense(num_nodes, input_dim=num_feature, activation=activation, kernel_initializer='he_uniform'))

    # Hidden layers
    for _ in range(num_layer - 1):
        model_e.add(Dense(num_nodes, activation=activation, kernel_initializer='he_uniform'))

    # Output layer based on target prediction quantity
    if target_type == 'energy':
        model_e.add(Dense(1, kernel_initializer='he_uniform'))
    elif target_type == "force":
        model_e.add(Dense(num_atom, kernel_initializer='he_uniform'))
    else:
        print("Error: unknown target quantity.")
        exit()

    model_e.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model_e


def def_normalization(
    x: np.ndarray, y: np.ndarray, target_type: str, xshift: float = 0, x_has_min: bool = False
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, float, float, float]:
    """
    Defines the normalization scaling parameters based on the target property being predicted.
    
    Args:
        x (np.ndarray): Input feature data.
        y (np.ndarray): Target data.
        target_type (str): The physical quantity to predict ('energy' or 'force').
        xshift (float): Shift parameter for global normalization (default: 0).
        x_has_min (bool): Whether to calculate a dynamic minimum for the inputs (default: False).
        
    Returns:
        Tuple: Scaling parameters containing (x_range, x_min, y_range, y_min, e_range, e_min).
    """
    if target_type == 'force':
        x_t, x_min, x_range = individual_normalize(x)
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
