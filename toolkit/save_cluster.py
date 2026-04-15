"""
Generating  projected interatomic distance, potential energy, projected force, kmeans cluster radius, average inertia for online ML server.
ex.
python save_cluster.py 3 H2O1
"""


from sklearn.cluster import MiniBatchKMeans
import time
import numpy as np
from scipy.spatial import KDTree
import os
from scipy.spatial import distance
import fragment_transform
from fragment_transform import atom_mass_mapping
from sys import argv

def Mini_batch(datapoints, weight, number_of_cluster, quantile=1.0):
    from incremental_clustering import get_cluster_radius
    time_ = time.time()
    clustering = MiniBatchKMeans(random_state=200, batch_size=10000, n_clusters=number_of_cluster
                                 , n_init=20).fit(datapoints, sample_weight=weight)
    radius_list = get_cluster_radius(datapoints, clustering, mode='max')
    radius = np.nanquantile(radius_list, quantile)
    inertia = clustering.inertia_ / len(datapoints)
    print('K-mean avg inertia:', inertia)
    print('max max radius: ', radius)
    print("\nTime for MiniBatchKmean clustering:", round(time.time() - time_, 2), "s.")
    return clustering.cluster_centers_, radius, inertia


def get_training_from_centroid(x, centroid):
    tree1 = KDTree(x)
    weight = None
    # print(centroid)
    dist1, train_order = tree1.query(centroid)
    return x[train_order], train_order


def save(x,r,path, id):
    np.savetxt(path+id+'_train.txt', x, fmt= '%.6f')
    np.savetxt(path+id+'_radius.txt', [r], fmt='%.6f')
    return


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


def input_data(xyz_path,atom_num_path,target_path,transfer_fragment=False):

    atom_num = np.loadtxt(atom_num_path)
    #print(atom_num)

    num_atom=len(atom_num[0])

    energy = np.loadtxt(target_path)

    xyz_a = np.loadtxt(xyz_path)
    xyz_a = xyz_a.reshape([int(np.size(xyz_a)/num_atom/3),num_atom,3])
    #print(xyz_a)

    num_dis = num_atom * (num_atom - 1) / 2
    dis_list = np.zeros([len(atom_num), int(num_dis)])
    for n in range(len(atom_num)):
        # print("axyz",axyz)
        atomic_numbers, xyz = atom_num[n],xyz_a[n]
        #
        if transfer_fragment:
            target_direction = 'z'
            ref = np.ones([num_atom, 3]) * 3
            atomic_mass = atom_mass_mapping(atomic_numbers)
            xyz, force, atomic_numbers, v, order = fragment_transform.transform_fragment(xyz, ref,
                                                                                            atomic_numbers, atomic_mass,
                                                                                            target_direction, ref)

        else:
            xyz = xyz[np.argsort(atomic_numbers)]  #### sort all coordinate by increasing atomic number
            atomic_numbers = np.sort(atomic_numbers)
            #### simple sorting dis list
        dis_matrix = distance.cdist(xyz, xyz)
        #if (n ==0 or n==1or n==2 or n==12) and False:
         #   print(dis_matrix)
        k = 0
        for i in range(len(dis_matrix) - 1):
            for j in range(i, len(dis_matrix) - 1):
                dis_list[n, k] = dis_matrix[i][j + 1]
                k += 1

    #print(energy)
    dis_list=np.delete(dis_list,np.where(energy==0),axis=0)
    energy = np.delete(energy,np.where(energy==0))

    return dis_list,energy


if __name__ == '__main__':
    sys_name = 'sz'
    num = argv[1]
    id = argv[2]
    frag_name = str(num)
    path = 'save_cluster/'
    num_atom = int(num)

    atom_num_path = sys_name + '/input/' + sys_name + '_atnum_' + str(num) + '.txt'
    xyz_path = sys_name + '/input/' + sys_name + '_xyz_' + str(num) + '.txt'
    ## delta E
    energy_d_path = sys_name + '/input/' + sys_name + 'e' + str(num) + 'diff.txt'
    energy_d_path_small = sys_name + '/input/+/' + sys_name + 'e' + str(num) + 'diff.txt'
    ### E level 1
    energy_1_path = sys_name + '/input/' + sys_name + '_' + str(frag_name) + '_high.txt'  ##[:,1]
    ###force element
    force_order = 0
    force_e_path = sys_name+'/input/' + sys_name + '_force_' + str(frag_name) + '_' + str(
                                    force_order) + '.txt'
    ### full force
    force_path = sys_name + '/input/+/' + 'f' + str(num)+'d_' + '.txt'
    ##energy = energy.reshape([len(atom_num), num_atom * 3])
    print('Loading data.')
    x, y = input_data(xyz_path, atom_num_path, energy_d_path_small, transfer_fragment=True)
    
    atn = np.loadtxt(atom_num_path)
    # xyz = np.loadtxt(xyz_path)

    xyz = np.loadtxt(xyz_path).reshape(-1)
    xyz = xyz.reshape([int(len(xyz)/3/num_atom),num_atom,3])


    #f = np.loadtxt(force_path)
    f = np.loadtxt(force_path).reshape(-1)
    f = f.reshape([int(len(f)/3/num_atom),num_atom,3])


    num_clu = int(len(y)*0.1)

    # x,xmin,x_range = global_normalize(x,shift=0,x_has_min=True)
    print('Clustering.')
    centroids, radius, inertia = Mini_batch(np.asarray(x),weight=None,number_of_cluster=num_clu, quantile=0.999)
    print('Saving.')
    x_train, train_order = get_training_from_centroid(x,centroids)
    #save(x_train ,radius, path, frag_name)
    np.savetxt(path+id+'_train.txt', x_train, fmt= '%.8f')
    np.savetxt(path+id+'_radius.txt', [radius], fmt='%.8f')
    np.savetxt(path+id+'_inertia.txt', [inertia], fmt='%0.12f')
    np.savetxt(path+id+'_train_xyz.txt', xyz[train_order].reshape([len(train_order),num_atom*3]), fmt= '%.8f')
    np.savetxt(path+id+'atn.txt', atn[train_order], fmt='%i')
    np.savetxt(path+id+'_e.txt', y[train_order], fmt='%.8f')
    np.savetxt(path+id+'_f.txt', f[train_order].reshape([len(train_order),num_atom*3]), fmt= '%.8f')
    np.savetxt(path+id+"_order.txt", train_order.reshape([len(train_order),1]), fmt='%i')
 
 
