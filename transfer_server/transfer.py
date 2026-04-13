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

warnings.simplefilter('ignore', UserWarning)
print = functools.partial(print, flush=True)
'''
input sz data - find average inertia - input 21mer data - divide 21mer data into slices of distance to sz centers - 
for every slice do kMean based on decided average inertia
'''

desktop_path = os.path.expanduser("~/Desktop")
test_path = desktop_path + "/research project/NN result/model_diff/test/"
model_path = r'C:\Users\Xiao\Desktop\research project\NN result\model_diff\6-31++g(d,p)\min shift\test model/'


# 10% model/'


def Mini_batch(datapoints, weight, number_of_cluster, print_time=True, init='k-means++'):
    time_ = time.time()
    clustering = (MiniBatchKMeans(random_state=200, batch_size=10000, n_clusters=number_of_cluster, n_init='auto',
                                  init=init).fit(datapoints, sample_weight=weight))
    if print_time:
        print("\nTime for MiniBatchKMean clustering:", round(time.time() - time_, 2), "s.")
    # print('avg inertia:', clustering.inertia_/len(datapoints),len(datapoints))
    # exit()
    return clustering


def increment():
    return


def distance_to_centers(x, query):
    tree = spatial.KDTree(x)
    dist, order = tree.query(query)
    return dist, order


def retrain(model_dict, frag_name, x, x_lag, y, y_lag, train_order, x_min, x_range, y_min, y_range,
            e_min, e_range, direction,new_epochs=100, mode='slice', batch=32):
    #atnum = len(x)
    #numatm = len(x[0]) * 2
    print(f"Initiate retraining for {frag_name}. Available model keys: {model_dict.keys()}")
    p = str(frag_name)

    x_p = (x - x_min) / x_range
    if len(x_lag) == 0:
        x_p_0 = [[]]
    else:
        x_p_0 = (x_lag - x_min) / x_range
    y_p, y_p_0 = (y - y_min) / y_range, (y_lag - y_min) / y_range

    if mode == 'slice':
        x_t = x_p[train_order]
        y_t = y_p[train_order]
    else:
        x_t = x_p
        y_t = y_p
    print(f'\n {frag_name} on {direction} y_t,y_p_0 shape: {y_t.shape}, {y_p_0.shape}\n')
    if len(x_lag) == 0:
        x_a = x_t
        y_a = y_t
    else:
        x_a = np.append(x_t, x_p_0, axis=0)
        y_a = np.append(y_t, y_p_0, axis=0)

    print(f'{len(x_t)} x_t[0]: ',x_t[0])
    print(f'{len(y_t)} y_t[0]: ',y_t[0])
    print(f'{len(y_t)} y[0]: ',y[0])
    print(f'y_min, y_range: ',y_min,y_range)
    if direction + "%s" % str(p) + "s" not in model_dict.keys():
        print(f'Error: key {direction + "%s" % str(p) + "s"} not found in model_dict in line 84. available keys: {model_dict.keys()}')
    model_dict[direction + "%s" % str(p) + "p"].summary()
    model_dict[direction+"%s" % str(p) + "p"].fit(x_t, y_t, epochs=new_epochs, verbose=0, callbacks=[tf.keras.callbacks.
                                        EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)],
                                        batch_size=batch)
    hist_0 = model_dict[direction+"%s" % str(p) + "p"].fit(x_a, y_a, epochs=10000, verbose=0, callbacks=[tf.keras.callbacks.
                                                 EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)],
                                                 batch_size=batch)

    #print(f'{frag_name} ', y_t.shape, y_t[0], x_t.shape, x_t[0])
    if direction == '':
        e_t = (y_t - model_dict[direction+"%s" % str(p) + "p"].predict(x_t, verbose=0).reshape(-1) - e_min) / e_range
        if len(x_lag) == 0:
            e_p_0 = [[]]
        else:
            e_p_0 = (y_p_0 - model_dict[direction+"%s" % str(p) + "p"].predict(x_p_0, verbose=0).reshape(-1) - e_min) / e_range
    else:
        e_t = (y_t - model_dict[direction+"%s" % str(p) + "p"].predict(x_t, verbose=0) - e_min) / e_range
        if len(x_lag) == 0:
            e_p_0 = [[]]
        else:
            e_p_0 = (y_p_0 - model_dict[direction+"%s" % str(p) + "p"].predict(x_p_0, verbose=0) - e_min) / e_range
    #print(e_t, e_p_0.shape, e_p_0[0])
    if len(x_lag) == 0:
        e_a = e_t
    else:
        e_a = np.append(e_t, e_p_0, axis=0)
    model_dict[direction+"%s" % str(p) + "s"].fit(x_t, e_t, epochs=new_epochs, verbose=0, callbacks=[tf.keras.callbacks.
                                        EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)])
    hist_e = model_dict[direction+"%s" % str(p) + "s"].fit(x_a, e_a, epochs=10000, verbose=0, callbacks=[tf.keras.callbacks.
                                                 EarlyStopping(monitor='loss', patience=20, min_delta=0.00001)])

    #p_e = model_dict[direction+"%s" % str(p) + "p"].predict(x_p, verbose=0) * y_range + y_min + (
    #        model_dict[direction+"%s" % str(p) + "s"].predict(x_p, verbose=0) * e_range + e_min) / 627.503
    #abs_err = np.absolute(p_e.reshape(-1) - y) * 627.503

    return


def online_expansion(x, treex, radius):
    tree = spatial.KDTree(treex)
    dist, order = tree.query(x)
    if dist[0] <= radius:
        # online_retrain()
        return None
    else:
        return x


def model_init(num_nodes, num_layer, x, activation, target_type='energy'):
    model = Sequential()
    ## available activation: "relu", "sigmoid", "softmax", "tanh" ......
    model.add(Dense(num_nodes, input_dim=len(x[0]), activation=activation, kernel_initializer='he_uniform'))
    for ii in range(num_layer):
        model.add(Dense(num_nodes, activation=activation, kernel_initializer='he_uniform'))

    if target_type == 'energy':
        model.add(Dense(1, kernel_initializer='he_uniform'))

    return model

'''
x_t, y_t: initial training set
x, y: transfer set
model_dict: initial model dict
radius: clustering radius cutoff
'''
def main(x_t, y_t, x, y, model_dict, param_dict, radius, avg_inertia, iD, num_atom, direction='',clustering_mode='sequential',
         partition='equal', mode='slice', inertia_m=4
         ):
    transfer_order = []
    transfer_order_list = []
    #x, y = input_data(sys_name, num_atom, transfer_fragment=trans_frag, basis='++') # target model data: w21h

    #model_dict = load_model([num_atom], model_path=model_path)
    x_range, x_min = param_dict[direction+"%s" % str(iD) + "px"]
    y_range, y_min = param_dict[direction+"%s" % str(iD) + "py"]
    e_range, e_min = param_dict[direction+"%s" % str(iD) + "sy"]

    #x_t = x_train[train_order] #initial model training point: sz
    #y_t = y_train[train_order]

    if len(x_t) == 0:
        dist2 = np.zeros(len(x))+radius+0.001
    else:
        tree2 = spatial.KDTree(x_t)
        dist2, test_order = tree2.query(x)  # 21mer with centroid for radius decision
    max_ord = np.where(dist2 == np.max(dist2))
    #print('max different geo, closest geo, avg geo:', x[max_ord], x_t[test_order[max_ord]], np.mean(x_t, axis=0))
    total = len(x)

    #err_list = test_model(x, y, sys_name, num_atom, model_dict, x_min, x_range, y_min, y_range, e_min, e_range)
    #mae_list, num_sample_list = [np.mean(np.abs(err_list))], [len(y_t)]
    train_size = len(y_t)

    if mode == 'slice':
        slice_list = []
        # slice_list = np.arrange(0, int(np.max(dist2) / radius) + 1)  # iterate through slices, single radius increment
        di2m = np.max(dist2)
        #print(f'Max distance: {di2m}')
        if partition == 'equal':
            slice_list = np.arange(radius, di2m + 1 * radius, 1 * radius)
            #print(slice_list)
        elif partition == 'kmeans':
            weight = None
            # print(dist2, len(dist2))
            p_center = np.sort(Mini_batch(dist2.reshape(-1, 1), weight, int(np.max(dist2) / radius / 2) + 1)
                               .cluster_centers_.reshape(-1))
            # print(dist2, len(dist2))
            slice_list = (p_center[1:] - p_center[0:-1]) / 2 + p_center[0:-1]
            # slice_list = np.append([0], p_center)
            # slice_list = (slice_list[1:] - slice_list[0:-1])
            # print(slice_list)
            if slice_list[0] > radius:
                slice_list = np.append([radius], slice_list)
            slice_list = np.append(slice_list, [di2m + 0.1])
            radius = None
        b_p = []
        #extrapolation_analysis_plot(sys_name, num_atom, dist2, err_list, 1, radius, num_sample_list,
        #                            slice_list, file_save_path=file_save_path, b_p=b_p)
        slice_1 = np.asarray(np.where((dist2 < slice_list[0]) == True))
        #np.savetxt(file_save_path + f'slice_1_order.txt',slice_1.reshape([np.size(slice_1), 1]), fmt="%i")
        for i in range(len(slice_list) - 1):
            time_ = time.time()
            slice_inertia = -1000
            #print(len(dist2), len(x), slice_list[i], slice_list[i + 1])

            #slice_order = np.logical_and(dist2 >= slice_list[i], dist2 <= slice_list[i + 1])
            slice_order = np.where((dist2 >= slice_list[i]) & (dist2 <= slice_list[i + 1]))[0]

            #np.savetxt(file_save_path + f'slice_{i + 2}_order.txt',
            #           np.asarray(np.where(slice_order == True)).reshape([np.sum(slice_order), 1]), fmt="%i")
            x_slice, y_slice = x[slice_order], y[slice_order]
            num_in_slice = len(x_slice)
            # print(slice_order, len(slice_order))
            if num_in_slice == 0:
                #num_sample_list.append(num_sample_list[-1])
                #mae_list.append(np.mean(np.abs(err_list)))
                #model_dict["%s" % str(num_atom) + "p"].save(
                #    test_path + "%s_primary_slice_%i.tf" % (str(num_atom), i + 2), overwrite=True)
                #model_dict["%s" % str(num_atom) + "s"].save(
                #    test_path + "%s_secondary_slice_%i.tf" % (str(num_atom), i + 2), overwrite=True)
                #np.savetxt(file_save_path + f'slice_{i + 2}_train_order.txt', [], fmt="%i")
                continue
            slice_num_cluster = int(train_size)
            if slice_num_cluster > num_in_slice or num_in_slice < 10 * train_size:
                slice_num_cluster = int(0.1 * num_in_slice)

            slice_train_order = None
            if clustering_mode == 'match':
                s_clustering = clustering_match(slice_inertia, avg_inertia, x_slice, num_in_slice, i, slice_num_cluster)

                tree3 = spatial.KDTree(x_slice)
                dist3, slice_train_order = tree3.query(s_clustering.cluster_centers_)
                # x_t_slice, y_t_slice = x_slice[slice_train_order], y_slice[slice_train_order]
            elif clustering_mode == 'sequential':
                slice_num_cluster = int(math.sqrt(num_in_slice))
                # print('x0',x[0],len(x[0]))
                centers, inertias = clustering_sequential(avg_inertia, x_slice, slice_num_cluster, i + 2,
                                                          0, len(x[0]), multiple_cutoff=inertia_m)
                print(f'slice {i + 2}, number of slice cluster: {len(centers)}/{num_in_slice} avg inertia: '
                      f'{np.sum(inertias) / num_in_slice}')
                tree3 = spatial.KDTree(x_slice)
                dist3, slice_train_order = tree3.query(centers)

            retrain(model_dict, iD, x_slice, x_t, y_slice, y_t, slice_train_order,
                        x_min,
                        x_range, y_min, y_range, e_min, e_range, direction, batch=30 * num_atom - 60)
            #np.savetxt(file_save_path + f'slice_{i + 2}_train_order.txt', slice_train_order, fmt="%i")
            if len(x_t) > 0:
                x_t, y_t = np.append(x_t, x_slice[slice_train_order], axis=0), np.append(y_t, y_slice[slice_train_order], axis=0)
            else:
                x_t, y_t = x_slice[slice_train_order], y_slice[slice_train_order]
            #err_list = test_model(x, y, sys_name, num_atom, model_dict, x_min, x_range, y_min, y_range, e_min, e_range)
            #mae_list.append(np.mean(np.abs(err_list)))
            #num_sample_list.append(num_sample_list[-1] + len(slice_train_order))
            #itr_accuracy_plot(sys_name, mae_list, num_sample_list, 'Number of training samples', total,
            #                  file_save_path=file_save_path)
            #extrapolation_analysis_plot(sys_name, num_atom, dist2, err_list, i + 2, radius, num_sample_list, slice_list,
            #                            file_save_path=file_save_path, b_p=b_p)
            transfer_order.extend(slice_order[slice_train_order].tolist())
            transfer_order_list.append(slice_order[slice_train_order].tolist())
            print(f'Time spend for slice {i + 2}: {round(time.time() - time_, 2)}s.')

    return transfer_order, transfer_order_list


def main_with_order(x_t, y_t, x, y, model_dict, param_dict, iD, num_atom, transfer_order_list, direction=''):

    x_range, x_min = param_dict[direction+"%s" % str(iD) + "px"]
    y_range, y_min = param_dict[direction+"%s" % str(iD) + "py"]
    e_range, e_min = param_dict[direction+"%s" % str(iD) + "sy"]

    for c, i in enumerate(transfer_order_list):
        x_slice = x[i]
        y_slice = y[i]
        time_ = time.time()
        retrain(model_dict, iD, x_slice, x_t, y_slice, y_t, np.arange(len(x_slice)),
                    x_min,
                    x_range, y_min, y_range, e_min, e_range, direction, batch=30 * num_atom - 60)

        if len(x_t) ==0:
            x_t, y_t = x_slice, y_slice
        else:
            x_t, y_t = np.append(x_t, x_slice, axis=0), np.append(y_t, y_slice, axis=0)

        print(f'Time spend for slice {c + 2}: {round(time.time() - time_, 2)}s.')


def clustering_match(slice_inertia, avg_inertia, x_slice, num_in_slice, i, slice_num_cluster):
    s_clustering = None
    n = 0
    w = 0
    d = 0
    while abs(slice_inertia - avg_inertia) > 0.2 * avg_inertia:  # find similar clustering for slices

        n += 1
        s_clustering = Mini_batch(x_slice, None, int(slice_num_cluster))
        slice_inertia = s_clustering.inertia_ / num_in_slice
        amplifier = 0.8 * (slice_inertia - avg_inertia) / avg_inertia

        print(f'slice {i + 2}, iteration {n}, number of slice cluster: {slice_num_cluster}/{num_in_slice}, '
              f'inertia: {slice_inertia}')

        delta_c = int(amplifier * slice_num_cluster)

        if abs(delta_c) == abs(d):
            delta_c = int(0.5 * delta_c)
        print(f'Modify number of cluster: {delta_c}')
        slice_num_cluster += delta_c
        if abs(delta_c) < 1:
            print('Warning: Kmeans fail to converge to the similar size.')
            # exit()
        if slice_num_cluster > num_in_slice:
            w += 1
            slice_num_cluster = int(0.8 * num_in_slice)
            if w == 2:
                slice_num_cluster = int(0.9 * num_in_slice)
            elif w == 3:
                slice_num_cluster = num_in_slice
        if w > 3:
            print("Warning: number of clusters have to be more than number of samples.")
            break
        if n > 10:
            slice_num_cluster -= int(0.5 * delta_c)
        if n > 20:
            print("Warning: Clustering matching fail to converge.")
            break

        d = delta_c
    return s_clustering


def clustering_sequential(avg_inertia, x_slice, slice_num_cluster, i, layer, x_0, multiple_cutoff=4):
    ll = layer + 1
    slice_size = len(x_slice)

    # if slice_size == 0:
    #   return np.empty((0, x_0)), np.empty([0, 1])
    # print(f'Slice {i}: creating layer {ll}, spawning {slice_num_cluster} clusters on {slice_size} samples')
    # print(x_0)
    center = np.empty([0, x_0])
    inertia_list = np.empty([0, 1])
    if slice_size == 2:
        slice_num_cluster = 2
    elif slice_size < slice_num_cluster:
        slice_num_cluster = round(0.51 * slice_size)
    if slice_size == 0:
        return None, None
    s_clustering = Mini_batch(x_slice, None, slice_num_cluster, print_time=False)
    # print('test ', slice_size, num_in_slice)
    # slice early termination
    '''
    slice_inertia = s_clustering.inertia_ / slice_size
    if slice_inertia <= multiple_cutoff * avg_inertia:

        # print(f'Slice {i}: finishing layer {ll} with {slice_num_cluster} centers on {slice_size} samples with avg '
        #       f'inertia {slice_inertia}.')

        return s_clustering.cluster_centers_, s_clustering.inertia_
    '''
    for ii in range(slice_num_cluster):
        sub_slice = x_slice[s_clustering.labels_ == ii]
        if len(sub_slice) == 0:
            continue
        sub_t_inertia = np.sum(np.square(sub_slice - s_clustering.cluster_centers_[ii]))
        sub_inertia = sub_t_inertia / len(sub_slice)
        if sub_inertia >= multiple_cutoff * avg_inertia:
            mult = round(sub_inertia / avg_inertia)
            center_, inertia_list_ = clustering_sequential(avg_inertia, sub_slice, mult, i, ll, x_0,
                                                           multiple_cutoff=multiple_cutoff)
            center = np.append(center, center_, axis=0)
            inertia_list = np.append(inertia_list, inertia_list_)
        else:
            # print(x_0,center,s_clustering.cluster_centers_[ii])
            center = np.append(center, [s_clustering.cluster_centers_[ii]], axis=0)
            inertia_list = np.append(inertia_list, sub_t_inertia)
    # print(f'Slice {i}: finishing layer {ll} with {len(center)} centers on {slice_size} samples with avg inertia '
    #       f'{np.sum(inertia_list)/slice_size}.')
    # print(center)
    return center, inertia_list


if __name__ == "__main__":

    print('')
