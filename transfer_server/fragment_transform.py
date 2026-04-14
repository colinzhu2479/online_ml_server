"""

Molecular Fragment Geometry Transformation Module.

We intend to introduce a consistent framework for description of molecular fragment geometries that arise from different molecular systems. Such a descriptor should allow for translational, rotational, and permutational invariance within the fragments, so that chemically identical fragments derived from different systems, or such fragments derived from different physical regions of the same system, have the same consistent mathematical description and thus can be compared with each other.

When training ML models on molecular data, the same physical configuration can appear in many equivalent representations depending on how the atoms are ordered or how the molecule is oriented in space. This module removes that ambiguity by:
* Sorting atoms by atomic number for a consistent ordering
* Centering coordinates on the center of mass
* Rotating the fragment into its principal inertia axis frame
* Resolving reflection ambiguity by aligning against a reference geometry

The result is a canonical representation where two physically identical configurations always map to the same input vector.

References:
    Xiao Zhu, S. S. Iyengar. Large Language Model-Type Architecture for High-Dimensional Molecular Potential Energy Surfaces. Physics Review X, 16, 011012 (2026). DOI: https://doi.org/10.1103/2qcy-8n8g
"""

import numpy as np
from scipy.spatial import distance
import math
import os
import matplotlib.pyplot as plt
import random
from typing import Tuple, Iterator, Optional


def center_of_mass_xyz(xyz: np.ndarray, atom_mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centers the coordinates of a fragment on its center of mass.
    
    Args:
        xyz (np.ndarray): Cartesian coordinates of the atoms (Shape: N x 3).
        atom_mass (np.ndarray): Masses of the atoms (Shape: N).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Centered Cartesian coordinates and the center of mass vector.
    """
    c_m = np.array([np.sum(xyz[:, i] * atom_mass) / np.sum(atom_mass) for i in range(3)])
    return xyz - c_m, c_m


def moment_of_innertia_axis(xyz: np.ndarray, atom_mass: np.ndarray) -> np.ndarray:
    """
    Computes the principal axes of inertia for the given coordinates.
    
    Args:
        xyz (np.ndarray): Centered Cartesian coordinates of the atoms (Shape: N x 3).
        atom_mass (np.ndarray): Masses of the atoms (Shape: N).
        
    Returns:
        np.ndarray: Matrix of eigenvectors (row vectors) representing the principal axes, 
                    sorted by descending eigenvalues.
    """
    Ixx = np.sum((xyz[:, 1] ** 2 + xyz[:, 2] ** 2) * atom_mass)
    Iyy = np.sum((xyz[:, 0] ** 2 + xyz[:, 2] ** 2) * atom_mass)
    Izz = np.sum((xyz[:, 1] ** 2 + xyz[:, 0] ** 2) * atom_mass)

    Ixy = -np.sum(xyz[:, 0] * xyz[:, 1] * atom_mass)
    Iyz = -np.sum(xyz[:, 2] * xyz[:, 1] * atom_mass)
    Ixz = -np.sum(xyz[:, 0] * xyz[:, 2] * atom_mass)

    I_m = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    w, v = np.linalg.eigh(I_m)  # in ascending order
    return v.T[::-1]  # row vector as basis with eigenvalues in decending order


def permute_by_type_and_dis(
    xyz: np.ndarray, force: np.ndarray, atom_num: np.ndarray, atom_mass: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sorts coordinates, forces, and masses by increasing atomic number to enforce 
    permutational invariance.
    
    Args:
        xyz (np.ndarray): Cartesian coordinates.
        force (np.ndarray): Atomic forces.
        atom_num (np.ndarray): Atomic numbers.
        atom_mass (np.ndarray): Atomic masses.
        
    Returns:
        Tuple: Sorted arrays (xyz, force, atom_num, atom_mass, sort_order_indices).
    """
    order = np.argsort(atom_num)
    xyz = xyz[order]  #### sort all coordinate by increasing atomic number
    force = force[order]
    atom_mass = atom_mass[order]
    atom_num = np.sort(atom_num)
    return xyz, force, atom_num, atom_mass, order


def comp_ref(
    xyz: np.ndarray, force: np.ndarray, atom_num: np.ndarray, atom_mass: np.ndarray, target_direction: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sets up a reference geometry for structure permutation mapping.
    
    Target direction stands for the permutation ordering for a specific force element. 
    'x' means the direction with the largest eigenvalue.
    
    Args:
        xyz (np.ndarray): Cartesian coordinates.
        force (np.ndarray): Atomic forces.
        atom_num (np.ndarray): Atomic numbers.
        atom_mass (np.ndarray): Atomic masses.
        target_direction (str): Axis identifier ('x', 'y', or 'z').
        
    Returns:
        Tuple: Processed (xyz, force, atom_num, principal_axes, original_order).
    """
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


def direction_order(target_direction: str) -> int:
    """
    Maps a string direction identifier to an index.
    
    Args:
        target_direction (str): 'x', 'y', or 'z'.
        
    Returns:
        int: Index 0, 1, or 2 respectively.
    """
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


def transform_fragment(
    xyz: np.ndarray, force: np.ndarray, atom_num: np.ndarray, atom_mass: np.ndarray, 
    target_direction: str, ref_xyz: np.ndarray, ref_v: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes a fragment by centering, rotating to principal axes, and aligning 
    against a reference geometry for a specific direction.
    
    Args:
        xyz (np.ndarray): Cartesian coordinates.
        force (np.ndarray): Atomic forces.
        atom_num (np.ndarray): Atomic numbers.
        atom_mass (np.ndarray): Atomic masses.
        target_direction (str): 'x', 'y', or 'z'.
        ref_xyz (np.ndarray): Reference geometry coordinates to align against.
        ref_v (Optional[np.ndarray]): Reference principal axes (default: None).
        
    Returns:
        Tuple: Transformed arrays (xyz, force, atom_num, principal_axes, index_order).
    """
    xyz, force, atom_num, atom_mass, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
    xyz, c_m = center_of_mass_xyz(xyz, atom_mass)
    v = moment_of_innertia_axis(xyz, atom_mass)

    xyz_s, force_s, atom_num, v, order_s = transform_fragment_3(
        np.asarray(xyz), np.asarray(force), atom_num, atom_mass, target_direction, v, np.asarray(order), ref_xyz, ref_v=None
    )
    return xyz_s, force_s, atom_num, v, order_s


def global_transform_fragment(
    xyz: np.ndarray, force: np.ndarray, atom_num: np.ndarray, atom_mass: np.ndarray, 
    ref_xyz: np.ndarray, ref_v: Optional[np.ndarray] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generator that transforms a fragment across all principal directions (x, y, z) 
    for reconstructing forces robustly.
    
    Args:
        xyz (np.ndarray): Cartesian coordinates.
        force (np.ndarray): Atomic forces.
        atom_num (np.ndarray): Atomic numbers.
        atom_mass (np.ndarray): Atomic masses.
        ref_xyz (np.ndarray): Reference geometry.
        ref_v (Optional[np.ndarray]): Reference axes.
        
    Yields:
        Tuple: Transformed arrays for each direction iteratively.
    """
    xyz1, force1, atom_num1, atom_mass1, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
    xyz2, c_m = center_of_mass_xyz(xyz1, atom_mass1)
    v = moment_of_innertia_axis(xyz2, atom_mass1)

    for target_direction in ['x', 'y', 'z']:
        xyz_s, force_s, atom_num, v, order_s = transform_fragment_3(
            np.array(xyz2), np.array(force1), atom_num1, atom_mass1,
            target_direction, v, np.array(order), ref_xyz, ref_v=None
        )
        yield xyz_s, force_s, atom_num, v, order_s


def transform_fragment_3(
    xyz: np.ndarray, force: np.ndarray, atom_num: np.ndarray, atom_mass: np.ndarray, 
    target_direction: str, v: np.ndarray, order: np.ndarray, ref_xyz: np.ndarray, ref_v: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core transformation engine. Projects coordinates into the principal axis frame 
    and resolves reflection ambiguity by comparing distance norms against `ref_xyz`.
    """
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
        ref_xyz[:, dir_ord] - xyz[:, dir_ord]
    )
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


def transform_fragment_2(
    xyz: np.ndarray, force: np.ndarray, atom_num: np.ndarray, atom_mass: np.ndarray, 
    target_direction: str, v: np.ndarray, order: np.ndarray, ref_xyz: np.ndarray, ref_v: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alternative/legacy implementation of `transform_fragment_3` utilizing explicit 

    if-else branching for targeted directions.
    """
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


def display(xyz: np.ndarray, ref_center: np.ndarray, ref_dir: np.ndarray) -> None:
    """
    Plots the molecular fragment in 3D space alongside a reference vector.
    

    Args:
        xyz (np.ndarray): Array containing 3D coordinates.
        ref_center (np.ndarray): The starting point of the reference direction vector.

        ref_dir (np.ndarray): The end point of the reference direction vector.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], marker='O')
    ax.plot([ref_center, ref_dir[0]])
    return


def atom_mass_mapping(atom_num: np.ndarray) -> np.ndarray:
    """

    Maps atomic numbers to roughly accurate atomic masses.
    
    Args:
        atom_num (np.ndarray): Array of atomic numbers.
        
    Returns:
        np.ndarray: Array of corresponding atomic masses.
    """
    mass = np.array([1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35.5, 40, 39, 40])
    return np.array([mass[int(atom_num[i] - 1)] for i in range(len(atom_num))])
