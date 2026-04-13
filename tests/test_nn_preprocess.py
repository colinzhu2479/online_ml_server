import pytest
import numpy as np
from scipy.spatial import distance
from transfer_server.nn_preprocess import *

@pytest.fixture
def water():
    xyz = np.array([
        [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
        [-2.481261367915496,  1.959085964931036,  2.293069675637073],
        [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
    ])
    atom_num = np.array([8, 1, 1])
    return xyz, atom_num
 
 
@pytest.fixture
def two_water_molecules():
    xyz = np.array([
        [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
        [-2.481261367915496,  1.959085964931036,  2.293069675637073],
        [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
        [ 2.9090009231356193,-1.635249157272597,  1.1466516333609733],
        [ 3.3888904269831794,-2.405690233783929,  0.815114885973314],
        [ 2.999338399227558, -1.6364081663645396, 2.1084487025789698],
    ])
    atom_num = np.array([8, 1, 1, 8, 1, 1])
    return xyz, atom_num
    
class TestInputProcess:

    def test_dis_list_length(self, water):
        xyz, atom_num = water
        dis_list, _ = input_process(xyz, atom_num)
        assert len(dis_list) == 3  # 3*(3-1)/2

    def test_dis_list_length_formula(self, two_water_molecules):
        """n=6 → 15 pairwise distances."""
        xyz, atom_num = two_water_molecules
        dis_list, _ = input_process(xyz, atom_num)
        assert len(dis_list) == 15  # 6*(6-1)/2

    def test_distances_positive_and_finite(self, water):
        xyz, atom_num = water
        dis_list, _ = input_process(xyz, atom_num)
        assert np.all(dis_list > 0) and np.all(np.isfinite(dis_list))

    def test_no_transform_branch_sorts_atoms(self, water):
        xyz, atom_num = water
        _, atom_num_out = input_process(xyz, atom_num, transfer_fragment=False)
        assert np.all(atom_num_out[:-1] <= atom_num_out[1:])

    def test_both_branches_same_sorted_distances(self, water):
        """Only meaningful cross-branch check: same distance set either way."""
        xyz, atom_num = water
        dis_with, _    = input_process(xyz, atom_num, transfer_fragment=True)
        dis_without, _ = input_process(xyz, atom_num, transfer_fragment=False)
        np.testing.assert_allclose(np.sort(dis_with), np.sort(dis_without), atol=1e-10)
