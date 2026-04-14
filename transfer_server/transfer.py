"""
Model Transfer Learning and Active Expansion Module.

This module provides the transfer learning support for adapting neural network 
models from small reference systems to larger target systems. It introduces methodologies 
for intelligently exploring the potential energy configuration space.

This is accomplished by taking a larger target system's data, dividing it into 
spatial "slices" based on distance to the reference system's centers, and using 
MiniBatch K-Means clustering to intelligently sample representative geometries for 
retraining.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time
from scipy import spatial
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import warnings
import functools
from typing import Optional, Tuple, List, Dict, Any, Union

warnings.simplefilter('ignore', UserWarning)
print = functools.partial(print, flush=True)


def Mini_batch(
    datapoints: np.ndarray, weight: Optional[np.ndarray], number_of_cluster: int, 
    print_time: bool = True, init: str = 'k-means++'
) -> MiniBatchKMeans:
    """
    Executes Mini-Batch K-Means clustering to partition the geometric data space.
    
    Args:
        datapoints (np.ndarray): Data features to cluster.
        weight (Optional[np.ndarray]): Sample weights for clustering.
        number_of_cluster (int): Target number of clusters (K).
        print_time (bool): Whether to print execution time (default: True).
        init (str): Initialization method (default: 'k-means++').
        
    Returns:
        MiniBatchKMeans: The fitted clustering model.
    """
    time_ = time.time()
    clustering = (MiniBatchKMeans(
        random_state=200, batch_size=10000, n_clusters=number_of_cluster, 
        n_init='auto', init=init
    ).fit(datapoints, sample_weight=weight))
    
    if print_time:
        print("\nTime for MiniBatchKMean clustering:", round(time.time() - time_, 2), "s.")
    return clustering


def distance_to_centers(x: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the nearest neighbor distances and indices using a KDTree.
    
    Args:
        x (np.ndarray): Reference data points to build the KDTree.
        query (np.ndarray): Query points to find distances for.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Distances and indices of the nearest neighbors.
    """
    tree = spatial.KDTree(x)
    dist, order = tree.query(query)
    return dist, order


def retrain(
    model_dict: Dict[str, Any], frag_name: str, x: np.ndarray, x_lag: Union[List, np.ndarray], 
    y: np.ndarray, y_lag: Union[List, np.ndarray], train_order: np.ndarray, 
    x_min: float, x_range: float, y_min: float, y_range: float, e_min: float, 
    e_range: float, direction: str, new_epochs: int = 100, mode: str = 'slice', batch: int = 32
) -> None:
    """
    Retrains primary and secondary models dynamically on new regions of the data space.
    
    Args:
        model_dict (Dict[str, Any]): Dictionary of keras models.
        frag_name (str): Identifier for the target fragment/model.
        x (np.ndarray): New input features to train on.
        x_lag (Union[List, np.ndarray]): Accumulated historic input features.
        y (np.ndarray): Target values for new inputs.
        y_lag (Union[List, np.ndarray]): Historic target values.
        train_order (np.ndarray): Indices representing the selected sub-sample (slice) to train on.
        x_min (float), x_range (float), y_min (float), y_range (float), e_min (float), e_range (float): Normalization parameters.
        direction (str): Projection direction ('x', 'y', 'z', or '').
        new_epochs (int): Epochs for active slice training.
        mode (str): Training mode (e.g., 'slice').
        batch (int): Batch size.
    """
    print(f"Initiate retraining for {frag_name}. Available model keys: {model_dict.keys()}")
    p = str(frag_name)

    # Normalize inputs and targets
    x_p = (x - x_min) / x_range
    x_p_0 = [[]] if len(x_lag) == 0 else (np.asarray(x_lag) - x_min) / x_range
    y_p = (y - y_min) / y_range
    y_p_0 = (np.asarray(y_lag) - y_min) / y_range

    if mode == 'slice':
        x_t = x_p[train_order]
        y_t = y_p[train_order]
    else:
        x_t, y_t = x_p, y_p

    print(f'\n {frag_name} on {direction} y_t,y_p_0 shape: {y_t.shape}, {np.asarray(y_p_0).shape}\n')
    
    # Append historic data to maintain memory of previous domains
    if len(x_lag) == 0:
        x_a, y_a = x_t, y_t
    else:
        x_a = np.append(x_t, x_p_0, axis=0)
        y_a = np.append(y_t, y_p_0, axis=0)

    if direction + f"{p}s" not in model_dict.keys():
        print(f'Error: key {direction + f"{p}s"} not found in model_dict. available keys: {model_dict.keys()}')

    # 1. Retrain the primary model
    model_dict[direction + f"{p}p"].fit(
        x_t, y_t, epochs=new_epochs, verbose=0, batch_size=batch,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)]
    )
    model_dict[direction + f"{p}p"].fit(
        x_a, y_a, epochs=10000, verbose=0, batch_size=batch,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)]
    )

    # 2. Calculate the residual errors (delta) for the secondary model
    e_t = (y_t - model_dict[direction + f"{p}p"].predict(x_t, verbose=0).reshape(-1 if direction == '' else y_t.shape) - e_min) / e_range
    
    if len(x_lag) == 0:
        e_p_0 = [[]]
    else:
        e_p_0 = (y_p_0 - model_dict[direction + f"{p}p"].predict(x_p_0, verbose=0).reshape(-1 if direction == '' else np.asarray(y_p_0).shape) - e_min) / e_range

    e_a = e_t if len(x_lag) == 0 else np.append(e_t, e_p_0, axis=0)

    # 3. Retrain the secondary model on the errors
    model_dict[direction + f"{p}s"].fit(
        x_t, e_t, epochs=new_epochs, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)]
    )
    model_dict[direction + f"{p}s"].fit(
        x_a, e_a, epochs=10000, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)]
    )


def online_expansion(x: np.ndarray, treex: np.ndarray, radius: float) -> Optional[np.ndarray]:
    """
    Checks if a configuration is within the known domain radius.
    """
    tree = spatial.KDTree(treex)
    dist, order = tree.query(x)
    return None if dist[0] <= radius else x


def model_init(num_nodes: int, num_layer: int, x: np.ndarray, activation: str, target_type: str = 'energy') -> Sequential:
    """
    Initializes a fresh Keras Sequential network.
    """
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=len(x[0]), activation=activation, kernel_initializer='he_uniform'))
    for _ in range(num_layer):
        model.add(Dense(num_nodes, activation=activation, kernel_initializer='he_uniform'))

    if target_type == 'energy':
        model.add(Dense(1, kernel_initializer='he_uniform'))
    return model


def main(
    x_t: np.ndarray, y_t: np.ndarray, x: np.ndarray, y: np.ndarray, 
    model_dict: Dict[str, Any], param_dict: Dict[str, Any], radius: float, 
    avg_inertia: float, iD: str, num_atom: int, direction: str = '',
    clustering_mode: str = 'sequential', partition: str = 'equal', mode: str = 'slice', inertia_m: float = 4
) -> Tuple[List[int], List[List[int]]]:
    """
    Main transfer learning loop. Partitions new data into spatial slices, selects 
    representative clusters, and iteratively retrains models to expand their domain.
    """
    transfer_order: List[int] = []
    transfer_order_list: List[List[int]] = []

    x_range, x_min = param_dict[direction + f"{iD}px"]
    y_range, y_min = param_dict[direction + f"{iD}py"]
    e_range, e_min = param_dict[direction + f"{iD}sy"]

    # Calculate distance of new data to existing training centroid KDTree
    if len(x_t) == 0:
        dist2 = np.zeros(len(x)) + radius + 0.001
    else:
        tree2 = spatial.KDTree(x_t)
        dist2, test_order = tree2.query(x)
        
    train_size = len(y_t)

    if mode == 'slice':
        di2m = np.max(dist2)
        # Determine the boundaries for slicing the configuration space
        if partition == 'equal':
            slice_list = np.arange(radius, di2m + 1 * radius, 1 * radius)
        elif partition == 'kmeans':
            p_center = np.sort(Mini_batch(dist2.reshape(-1, 1), None, int(np.max(dist2) / radius / 2) + 1).cluster_centers_.reshape(-1))
            slice_list = (p_center[1:] - p_center[0:-1]) / 2 + p_center[0:-1]
            if slice_list[0] > radius:
                slice_list = np.append([radius], slice_list)
            slice_list = np.append(slice_list, [di2m + 0.1])

        for i in range(len(slice_list) - 1):
            time_ = time.time()
            slice_order = np.where((dist2 >= slice_list[i]) & (dist2 <= slice_list[i + 1]))[0]
            x_slice, y_slice = x[slice_order], y[slice_order]
            num_in_slice = len(x_slice)

            if num_in_slice == 0: continue
            
            slice_num_cluster = int(train_size)
            if slice_num_cluster > num_in_slice or num_in_slice < 10 * train_size:
                slice_num_cluster = int(0.1 * num_in_slice)

            slice_train_order = None
            if clustering_mode == 'match':
                s_clustering = clustering_match(-1000, avg_inertia, x_slice, num_in_slice, i, slice_num_cluster)
                tree3 = spatial.KDTree(x_slice)
                dist3, slice_train_order = tree3.query(s_clustering.cluster_centers_)

            elif clustering_mode == 'sequential':
                slice_num_cluster = int(math.sqrt(num_in_slice))
                centers, inertias = clustering_sequential(avg_inertia, x_slice, slice_num_cluster, i + 2, 0, len(x[0]), multiple_cutoff=inertia_m)
                print(f'slice {i + 2}, number of slice cluster: {len(centers)}/{num_in_slice} avg inertia: {np.sum(inertias) / num_in_slice}')
                tree3 = spatial.KDTree(x_slice)
                dist3, slice_train_order = tree3.query(centers)

            # Trigger retraining on the discovered representative samples
            retrain(model_dict, iD, x_slice, x_t, y_slice, y_t, slice_train_order, x_min, x_range, y_min, y_range, e_min, e_range, direction, batch=30 * num_atom - 60)

            # Expand the known domain
            if len(x_t) > 0:
                x_t = np.append(x_t, x_slice[slice_train_order], axis=0)
                y_t = np.append(y_t, y_slice[slice_train_order], axis=0)
            else:
                x_t, y_t = x_slice[slice_train_order], y_slice[slice_train_order]

            transfer_order.extend(slice_order[slice_train_order].tolist())
            transfer_order_list.append(slice_order[slice_train_order].tolist())
            print(f'Time spend for slice {i + 2}: {round(time.time() - time_, 2)}s.')

    return transfer_order, transfer_order_list


def main_with_order(
    x_t: np.ndarray, y_t: np.ndarray, x: np.ndarray, y: np.ndarray, 
    model_dict: Dict[str, Any], param_dict: Dict[str, Any], iD: str, 
    num_atom: int, transfer_order_list: List[List[int]], direction: str = ''
) -> None:
    """
    Executes sequential retraining using a pre-determined list of data index slices.
    """
    x_range, x_min = param_dict[direction + f"{iD}px"]
    y_range, y_min = param_dict[direction + f"{iD}py"]
    e_range, e_min = param_dict[direction + f"{iD}sy"]

    for c, i in enumerate(transfer_order_list):
        x_slice = x[i]
        y_slice = y[i]
        time_ = time.time()
        retrain(model_dict, iD, x_slice, x_t, y_slice, y_t, np.arange(len(x_slice)), x_min, x_range, y_min, y_range, e_min, e_range, direction, batch=30 * num_atom - 60)

        if len(x_t) == 0:
            x_t, y_t = x_slice, y_slice
        else:
            x_t = np.append(x_t, x_slice, axis=0)
            y_t = np.append(y_t, y_slice, axis=0)

        print(f'Time spend for slice {c + 2}: {round(time.time() - time_, 2)}s.')


def clustering_match(
    slice_inertia: float, avg_inertia: float, x_slice: np.ndarray, 
    num_in_slice: int, i: int, slice_num_cluster: int
) -> MiniBatchKMeans:
    """
    Iteratively adjusts K-Means clustering parameters to match a target average inertia.
    """
    s_clustering = None
    n, w, d = 0, 0, 0
    
    while abs(slice_inertia - avg_inertia) > 0.2 * avg_inertia:
        n += 1
        s_clustering = Mini_batch(x_slice, None, int(slice_num_cluster))
        slice_inertia = s_clustering.inertia_ / num_in_slice
        amplifier = 0.8 * (slice_inertia - avg_inertia) / avg_inertia

        print(f'slice {i + 2}, iteration {n}, number of slice cluster: {slice_num_cluster}/{num_in_slice}, inertia: {slice_inertia}')

        delta_c = int(amplifier * slice_num_cluster)
        if abs(delta_c) == abs(d): delta_c = int(0.5 * delta_c)
        
        slice_num_cluster += delta_c
        
        if slice_num_cluster > num_in_slice:
            w += 1
            slice_num_cluster = int(0.8 * num_in_slice)
            if w == 2: slice_num_cluster = int(0.9 * num_in_slice)
            elif w == 3: slice_num_cluster = num_in_slice
            
        if w > 3 or n > 20: break
        if n > 10: slice_num_cluster -= int(0.5 * delta_c)
        d = delta_c
        
    return s_clustering


def clustering_sequential(
    avg_inertia: float, x_slice: np.ndarray, slice_num_cluster: int, 
    i: int, layer: int, x_0: int, multiple_cutoff: float = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recursively clusters data segments until the sub-inertia falls below a multiple of the target average.
    """
    slice_size = len(x_slice)
    center = np.empty([0, x_0])
    inertia_list = np.empty([0, 1])
    
    if slice_size == 2: slice_num_cluster = 2
    elif slice_size < slice_num_cluster: slice_num_cluster = round(0.51 * slice_size)
    if slice_size == 0: return None, None
    
    s_clustering = Mini_batch(x_slice, None, slice_num_cluster, print_time=False)

    for ii in range(slice_num_cluster):
        sub_slice = x_slice[s_clustering.labels_ == ii]
        if len(sub_slice) == 0: continue
        
        sub_t_inertia = np.sum(np.square(sub_slice - s_clustering.cluster_centers_[ii]))
        sub_inertia = sub_t_inertia / len(sub_slice)
        
        if sub_inertia >= multiple_cutoff * avg_inertia:
            mult = round(sub_inertia / avg_inertia)
            center_, inertia_list_ = clustering_sequential(
                avg_inertia, sub_slice, mult, i, layer + 1, x_0, multiple_cutoff=multiple_cutoff
            )
            center = np.append(center, center_, axis=0)
            inertia_list = np.append(inertia_list, inertia_list_)
        else:
            center = np.append(center, [s_clustering.cluster_centers_[ii]], axis=0)
            inertia_list = np.append(inertia_list, sub_t_inertia)

    return center, inertia_list

if __name__ == "__main__":
    print('')
