import numpy as np
from scipy.spatial import distance
import math
import os
import matplotlib.pyplot as plt
import random


def center_of_mass_xyz(xyz, atom_mass):
    c_m = np.array([np.sum(xyz[:, i] * atom_mass) / np.sum(atom_mass) for i in range(3)])
    return xyz - c_m, c_m


def moment_of_innertia_axis(xyz, atom_mass):
    Ixx = np.sum((xyz[:, 1] ** 2 + xyz[:, 2] ** 2) * atom_mass)
    Iyy = np.sum((xyz[:, 0] ** 2 + xyz[:, 2] ** 2) * atom_mass)
    Izz = np.sum((xyz[:, 1] ** 2 + xyz[:, 0] ** 2) * atom_mass)

    Ixy = -np.sum(xyz[:, 0] * xyz[:, 1] * atom_mass)
    Iyz = -np.sum(xyz[:, 2] * xyz[:, 1] * atom_mass)
    Ixz = -np.sum(xyz[:, 0] * xyz[:, 2] * atom_mass)

    I_m = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    w, v = np.linalg.eigh(I_m)  # in ascending order
    return v.T[::-1]  # row vector as basis with eigenvalues in decending order


'''target direction stands for the permutation ordering for a specific force element. X means the direction with
the largest eigenvalue
'''
def permute_by_type_and_dis(xyz, force, atom_num, atom_mass):
    order = np.argsort(atom_num)
    xyz = xyz[order]  #### sort all coordinate by increasing atomic number
    force = force[order]
    atom_mass = atom_mass[order]
    atom_num = np.sort(atom_num)
    return xyz, force, atom_num, atom_mass, order


'''when used for set up reference geometry for all other structure permutation'''


def comp_ref(xyz, force, atom_num, atom_mass, target_direction):
    xyz, force, atom_num, atom_mass, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
    xyz, c_m = center_of_mass_xyz(xyz, atom_mass)
    v = moment_of_innertia_axis(xyz, atom_mass)

    if target_direction == 'x':

        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i

            xyz_temp = xyz[atm_pos][np.argsort((v[0] @ xyz.T)[atm_pos])]
            force_temp = force[atm_pos][np.argsort((v[0] @ xyz.T)[atm_pos])]

            xyz[atm_pos] = xyz_temp
            force[atm_pos] = force_temp

    elif target_direction == 'y':

        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][np.argsort((v[1] @ force.T)[atm_pos])]
            force_temp = force[atm_pos][np.argsort((v[1] @ force.T)[atm_pos])]
            xyz[atm_pos] = xyz_temp
            force[atm_pos] = force_temp
    elif target_direction == 'z':

        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][np.argsort((v[2] @ xyz.T)[atm_pos])]
            force_temp = force[atm_pos][np.argsort((v[2] @ xyz.T)[atm_pos])]
            xyz[atm_pos] = xyz_temp
            force[atm_pos] = force_temp
    else:
        print('Error: Unknown target direction.')
        exit()

    return xyz, force, atom_num, v, order


def direction_order(target_direction):
    if target_direction == 'x':
        dir_ord = 0
    elif target_direction == 'y':
        dir_ord = 1
    elif target_direction == 'z':
        dir_ord = 2
    else:
        print('Error: unknow direction.')
        exit()
    return dir_ord


def transform_fragment(xyz, force, atom_num, atom_mass, target_direction, ref_xyz, ref_v=None):
    xyz, force, atom_num, atom_mass, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
    xyz, c_m = center_of_mass_xyz(xyz, atom_mass)
    v = moment_of_innertia_axis(xyz, atom_mass)

    xyz_s, force_s, atom_num, v, order_s = transform_fragment_3(np.asarray(xyz), np.asarray(force), atom_num, atom_mass, target_direction, v, np.asarray(order),ref_xyz, ref_v=None)
    return xyz_s, force_s, atom_num, v, order_s


'''transform fragment by all directions for reconstructing forces'''
def global_transform_fragment(xyz, force, atom_num, atom_mass, ref_xyz, ref_v=None):
    xyz1, force1, atom_num1, atom_mass1, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
    xyz2, c_m = center_of_mass_xyz(xyz1, atom_mass1)
    v = moment_of_innertia_axis(xyz2, atom_mass1)

    for target_direction in ['x', 'y', 'z']:
        xyz_s, force_s, atom_num, v, order_s = transform_fragment_3(np.array(xyz2), np.array(force1), atom_num1, atom_mass1,
                                                                  target_direction, v, np.array(order), ref_xyz,
                                                                  ref_v=None)

        yield xyz_s, force_s, atom_num, v, order_s


def transform_fragment_3(xyz, force, atom_num, atom_mass, target_direction, v, order, ref_xyz, ref_v=None):
    dir_ord = direction_order(target_direction)

    for i in dict.fromkeys(atom_num).keys():
        atm_pos = atom_num == i
        proj_order = np.argsort((v[dir_ord] @ xyz.T)[atm_pos])
        xyz_temp = xyz[atm_pos][proj_order] @ v.T
        order_temp = order[atm_pos][proj_order]
        force_temp = force[atm_pos][proj_order]
        xyz[atm_pos] = xyz_temp
        order[atm_pos] = order_temp
        force[atm_pos] = force_temp

    t = np.linalg.norm(ref_xyz[:, dir_ord] - -1 * xyz[::-1][:, dir_ord]) - np.linalg.norm(
        ref_xyz[:, dir_ord] - xyz[:, dir_ord])
    ordd = int(math.copysign(1, t))
    v[dir_ord] = ordd * v[dir_ord]
    for i in dict.fromkeys(atom_num).keys():
        atm_pos = atom_num == i
        xyz_temp = xyz[atm_pos][::ordd]
        xyz_temp[:, dir_ord] = ordd * xyz_temp[:, dir_ord]
        xyz[atm_pos] = xyz_temp

        order_temp = order[atm_pos][::ordd]
        order[atm_pos] = order_temp

        force_temp = force[atm_pos][::ordd]
        force[atm_pos] = force_temp

    return xyz, force, atom_num, v, order


def transform_fragment_2(xyz, force, atom_num, atom_mass, target_direction, v, order, ref_xyz, ref_v=None):

    if target_direction == 'x':


        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i

            xyz_temp = xyz[atm_pos][np.argsort((v[0] @ xyz.T)[atm_pos])] @ v.T
            order_temp = order[atm_pos][np.argsort((v[0] @ xyz.T)[atm_pos])]
            force_temp = force[atm_pos][np.argsort((v[0] @ xyz.T)[atm_pos])]

            xyz[atm_pos] = xyz_temp
            order[atm_pos] = order_temp
            force[atm_pos] = force_temp

        t = np.linalg.norm(ref_xyz[:, 0] - -1 * xyz[::-1][:, 0]) - np.linalg.norm(
            ref_xyz[:, 0] - xyz[:, 0])
        ord = int(math.copysign(1, (t)))
        v[0] = ord * v[0]
        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][::ord]
            xyz_temp[:, 0] = ord * xyz_temp[:, 0]
            xyz[atm_pos] = xyz_temp

            order_temp = order[atm_pos][::ord]
            order[atm_pos] = order_temp

            force_temp = force[atm_pos][::ord]

            force[atm_pos] = force_temp

    elif target_direction == 'y':

        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][np.argsort((v[1] @ xyz.T)[atm_pos])] @ v.T
            order_temp = order[atm_pos][np.argsort((v[1] @ xyz.T)[atm_pos])]
            force_temp = force[atm_pos][np.argsort((v[1] @ xyz.T)[atm_pos])]

            xyz[atm_pos] = xyz_temp
            order[atm_pos] = order_temp
            force[atm_pos] = force_temp

        t = np.linalg.norm(ref_xyz[:, 1] - -1 * xyz[::-1][:, 1]) - np.linalg.norm(
            ref_xyz[:, 1] - xyz[:, 1])
        ord = int(math.copysign(1, (t)))
        v[1] = ord * v[1]
        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][::ord]
            xyz_temp[:, 1] = ord * xyz_temp[:, 1]
            xyz[atm_pos] = xyz_temp

            order_temp = order[atm_pos][::ord]
            order[atm_pos] = order_temp

            force_temp = force[atm_pos][::ord]

            force[atm_pos] = force_temp
    elif target_direction == 'z':

        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][np.argsort((v[2] @ xyz.T)[atm_pos])] @ v.T
            order_temp = order[atm_pos][np.argsort((v[2] @ xyz.T)[atm_pos])]
            force_temp = force[atm_pos][np.argsort((v[2] @ xyz.T)[atm_pos])]

            xyz[atm_pos] = xyz_temp
            order[atm_pos] = order_temp
            force[atm_pos] = force_temp

        t = np.linalg.norm(ref_xyz[:, 2] - -1 * xyz[::-1][:, 2]) - np.linalg.norm(
            ref_xyz[:, 2] - xyz[:, 2])
        ord = int(math.copysign(1, (t)))
        v[2] = ord * v[2]
        for i in range(1, int(np.max(atom_num)) + 1):
            atm_pos = atom_num == i
            xyz_temp = xyz[atm_pos][::ord]
            xyz_temp[:, 2] = ord * xyz_temp[:, 2]
            xyz[atm_pos] = xyz_temp

            order_temp = order[atm_pos][::ord]
            order[atm_pos] = order_temp

            force_temp = force[atm_pos][::ord]

            force[atm_pos] = force_temp
    else:
        print('Error: Unknown target direction.')
        exit()

    return xyz, force, atom_num, v, order


def display(xyz, ref_center, ref_dir):
    fig = plt.figure()
    ax = fig.add_subplot(prokection='3d')
    ax.scatter(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], marker='O')
    ax.plot([ref_center, ref_dir[0]])
    return


def atom_mass_mapping(atom_num):
    mass = np.array([1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35.5, 40, 39, 40])
    # print(atom_num)
    return np.array([mass[int(atom_num[i] - 1)] for i in range(len(atom_num))])

