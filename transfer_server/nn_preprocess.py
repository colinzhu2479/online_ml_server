"""
This script is only for importing geometry information for training set spanning. Force and energy is not included and so the fragment_transform method use a place holder ref for force input.
"""
import numpy as np
from transfer_server import fragment_transform

def input_process(xyz_a, atom_num, transfer_fragment=True):
    from scipy.spatial import distance
    num_atom = len(atom_num)
    num_dis = num_atom * (num_atom - 1) / 2
    dis_list = np.zeros([int(num_dis)])

    atomic_numbers, xyz = atom_num, xyz_a
    #
    if transfer_fragment:
        target_direction = 'z'
        ref = np.ones([num_atom, 3]) * 3
        atomic_mass = fragment_transform.atom_mass_mapping(atomic_numbers)
        xyz, force, atomic_numbers, v, order = fragment_transform.transform_fragment(xyz, ref,
                                                                                     atomic_numbers, atomic_mass,
                                                                                     target_direction, ref)
    else:
        xyz = xyz[np.argsort(atomic_numbers)]  #### sort all coordinate by increasing atomic number
        atomic_numbers = np.sort(atomic_numbers)
        #### simple sorting dis list
    dis_matrix = distance.cdist(xyz, xyz)

    k = 0
    for i in range(len(dis_matrix) - 1):
        for j in range(i, len(dis_matrix) - 1):
            dis_list[k] = dis_matrix[i][j + 1]
            k += 1

    return dis_list, atomic_numbers


