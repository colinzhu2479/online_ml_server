"""
Transfer Learning Data Preparation Module.

This module is part of a machine learning pipeline for molecular potential energy 
surfaces and atomic forces. It handles the loading, parsing, and geometric 
transformation of training data, preparing it for transfer learning tasks. 
By converting Cartesian coordinates and forces into invariant distance matrices 
and projected components, it enables the model to adapt from small reference systems 
to larger target molecular systems.
"""

import numpy as np
import fragment_transform
from scipy.spatial import distance
from typing import Dict, List, Tuple, Any


def load_force_input(frag_name: str, file_path: str) -> Dict[str, Any]:
    """
    Loads structural geometries, forces, and energies from text files and processes 
    them into a formatted dictionary for training.
    
    Args:
        frag_name (str): The prefix/identifier of the fragment files to load.
        file_path (str): The directory path containing the data files.
        
    Returns:
        Dict[str, Any]: A dictionary containing processed distance inputs ('d') and 
                        force targets ('f') for x, y, and z directions, plus total energies ('e').
    """
    train_dict: Dict[str, Any] = {}
    
    # Initialize dictionary keys for each direction's distances and forces
    for direction in ['x', 'y', 'z']:
        for ty in ['d', 'f']:
            train_dict[direction + ty] = []
            
    # Load atomic numbers and determine the number of atoms
    atom_num = np.loadtxt(file_path + f'{frag_name}atn.txt', ndmin=2)
    num_atom = len(atom_num[0])

    # Load and reshape coordinates (N_samples x N_atoms x 3)
    xyz_a = np.loadtxt(file_path + f'{frag_name}_train_xyz.txt').reshape(-1)
    xyz_a = xyz_a.reshape([int(len(xyz_a) / 3 / num_atom), num_atom, 3])

    # Load and reshape force vectors
    force = np.loadtxt(file_path + f'{frag_name}_f.txt').reshape(-1)
    force = force.reshape([int(len(force) / 3 / num_atom), num_atom, 3])

    # Load energy labels
    train_dict['e'] = np.loadtxt(file_path + f'{frag_name}_e.txt').reshape(-1).tolist()

    # Process each configuration for all three principal directions
    for n in range(len(atom_num)):
        for direction in ['x', 'y', 'z']:
            dis_list, force_t = get_force_io(xyz_a[n], atom_num[n], num_atom, force[n], direction)
            train_dict[direction + 'd'].append(dis_list)
            train_dict[direction + 'f'].append(force_t)
            
    return train_dict


def get_force_io(
    xyz_a: np.ndarray, atom_num: np.ndarray, num_atom: int, force: np.ndarray, target_direction: str
) -> Tuple[List[float], List[float]]:
    """
    Transforms a single molecular configuration into a rotationally/translationally 
    invariant distance matrix and extracts the target force components.
    
    Args:
        xyz_a (np.ndarray): Cartesian coordinates of the molecule.
        atom_num (np.ndarray): Array of atomic numbers.
        num_atom (int): Total number of atoms.
        force (np.ndarray): Atomic force vectors.
        target_direction (str): The axis of projection ('x', 'y', or 'z').
        
    Returns:
        Tuple[List[float], List[float]]: 
            - Flat list of pairwise distances (lower dimensional subsystem space).
            - List of transformed force components along the target direction.
    """
    num_dis = int(num_atom * (num_atom - 1) / 2)
    dis_list = np.zeros([num_dis])

    # Reference geometry setup for reflection ambiguity resolution
    ref = np.ones([num_atom, 3]) * 3

    atomic_numbers = atom_num
    xyz = xyz_a
    atomic_mass = fragment_transform.atom_mass_mapping(atomic_numbers)

    # Standardize the fragment geometry
    xyz, force_t, atomic_numbers, v, order = fragment_transform.transform_fragment(
        xyz, np.asarray(force), atomic_numbers, atomic_mass, target_direction, ref
    )

    # Calculate pairwise distance matrix to serve as invariant features
    dis_matrix = distance.cdist(xyz, xyz)
    k = 0
    for i in range(len(dis_matrix) - 1):
        for j in range(i, len(dis_matrix) - 1):
            dis_list[k] = dis_matrix[i][j + 1]
            k += 1
            
    # Project forces onto the target principal axis
    force_t = np.array([np.matmul(v, force_t[iii]) for iii in range(num_atom)])[:, fragment_transform.direction_order(target_direction)]

    return dis_list.tolist(), force_t.tolist()


def dimer_test() -> None:
    """
    Test harness demonstrating the transformation and distance extraction 
    on a sample water dimer system.
    """
    num_atom = 6
    num_dis = int(num_atom * (num_atom - 1) / 2)
    dis_list = np.zeros([num_dis])
    
    xyz = np.array([
        -2.1522976348577143, 1.935813831410094, 1.3844868868883233,
        -2.481261367915496, 1.959085964931036, 2.293069675637073,
        -2.607829012546703, 2.6389455523580923, 0.9029163487370644,
        2.9090009231356193, -1.635249157272597, 1.1466516333609733,
        3.3888904269831794, -2.405690233783929, 0.815114885973314,
        2.999338399227558, -1.6364081663645396, 2.1084487025789698
    ]).reshape(6, 3)
    
    force = np.array([
        -2.1522976348577143, 1.935813831410094, 1.3844868868883233,
        -2.481261367915496, 1.959085964931036, 2.293069675637073,
        -2.607829012546703, 2.6389455523580923, 0.9029163487370644,
        2.9090009231356193, -1.635249157272597, 1.1466516333609733,
        3.3888904269831794, -2.405690233783929, 0.815114885973314,
        2.999338399227558, -1.6364081663645396, 2.1084487025789698
    ]).reshape(6, 3)
    
    atomic_numbers = np.array([8, 1, 1, 8, 1, 1])
    atomic_mass = np.array([16, 1, 1, 16, 1, 1])
    target_direction = 'z'
    ref = np.ones([6, 3])
    
    xyz, force_t, atomic_numbers, v, order = fragment_transform.transform_fragment(
        xyz, np.asarray(force), atomic_numbers, atomic_mass, target_direction, ref
    )

    dis_matrix = distance.cdist(xyz, xyz)
    k = 0
    for i in range(len(dis_matrix) - 1):
        for j in range(i, len(dis_matrix) - 1):
            dis_list[k] = dis_matrix[i][j + 1]
            k += 1
            
    force_t = np.array([np.matmul(v, force_t[iii]) for iii in range(num_atom)])[:, fragment_transform.direction_order(target_direction)]
    
    print(dis_list.tolist())
    print(force_t.tolist())

if __name__ == "__main__":
    dimer_test()
