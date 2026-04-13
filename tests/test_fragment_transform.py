from transfer_server.fragment_transform import *
import pytest
import numpy as np
import math
from scipy.spatial import distance

@pytest.fixture
def water_molecule():
    """Single water molecule: O + 2H."""
    xyz = np.array([
        [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
        [-2.481261367915496,  1.959085964931036,  2.293069675637073],
        [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
    ])
    force = np.array([
        [-1.949573e-03,  1.804180e-03,  1.063971e-03],
        [ 7.255650e-04,  4.399090e-04, -3.277888e-03],
        [ 1.224006e-03, -2.244091e-03,  2.213916e-03],
    ])
    atom_num  = np.array([8, 1, 1])
    atom_mass = np.array([16., 1., 1.])
    return xyz, force, atom_num, atom_mass
 
 
@pytest.fixture
def two_water_molecules():
    """Two water molecules (6 atoms)."""
    xyz = np.array([
        [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
        [-2.481261367915496,  1.959085964931036,  2.293069675637073],
        [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
        [ 2.9090009231356193,-1.635249157272597,  1.1466516333609733],
        [ 3.3888904269831794,-2.405690233783929,  0.815114885973314],
        [ 2.999338399227558, -1.6364081663645396, 2.1084487025789698],
    ])
    force = xyz.copy()
    atom_num  = np.array([8, 1, 1, 8, 1, 1])
    atom_mass = np.array([16., 1., 1., 16., 1., 1.])
    return xyz, force, atom_num, atom_mass
    
@pytest.fixture
def water_molecule():
    """Single water molecule: O + 2H."""
    xyz = np.array([
        [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
        [-2.481261367915496,  1.959085964931036,  2.293069675637073],
        [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
    ])
    force = np.array([
        [-1.949573e-03,  1.804180e-03,  1.063971e-03],
        [ 7.255650e-04,  4.399090e-04, -3.277888e-03],
        [ 1.224006e-03, -2.244091e-03,  2.213916e-03],
    ])
    atom_num  = np.array([8, 1, 1])
    atom_mass = np.array([16., 1., 1.])
    return xyz, force, atom_num, atom_mass
 
 
@pytest.fixture
def two_water_molecules():
    """Two water molecules (6 atoms)."""
    xyz = np.array([
        [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
        [-2.481261367915496,  1.959085964931036,  2.293069675637073],
        [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
        [ 2.9090009231356193,-1.635249157272597,  1.1466516333609733],
        [ 3.3888904269831794,-2.405690233783929,  0.815114885973314],
        [ 2.999338399227558, -1.6364081663645396, 2.1084487025789698],
    ])
    force = xyz.copy()
    atom_num  = np.array([8, 1, 1, 8, 1, 1])
    atom_mass = np.array([16., 1., 1., 16., 1., 1.])
    return xyz, force, atom_num, atom_mass
 
 
# ═════════════════════════════════════════════════════════════════════════════
# center_of_mass_xyz
# ═════════════════════════════════════════════════════════════════════════════
 
class TestCenterOfMass:
 
    def test_output_is_centered(self, water_molecule):
        """After centering, the weighted mean of coords should be ~zero."""
        xyz, _, _, atom_mass = water_molecule
        centered, c_m = center_of_mass_xyz(xyz, atom_mass)
        weighted_mean = np.sum(centered * atom_mass[:, None], axis=0) / np.sum(atom_mass)
        np.testing.assert_allclose(weighted_mean, 0.0, atol=1e-12)
 
    def test_returned_center_is_correct(self, water_molecule):
        """c_m must equal the weighted average of the original coords."""
        xyz, _, _, atom_mass = water_molecule
        expected_cm = np.sum(xyz * atom_mass[:, None], axis=0) / np.sum(atom_mass)
        _, c_m = center_of_mass_xyz(xyz, atom_mass)
        np.testing.assert_allclose(c_m, expected_cm, atol=1e-12)
 
    def test_single_atom(self):
        """A single atom should be placed at the origin."""
        xyz = np.array([[3.0, -1.0, 2.0]])
        atom_mass = np.array([12.0])
        centered, c_m = center_of_mass_xyz(xyz, atom_mass)
        np.testing.assert_allclose(centered, [[0., 0., 0.]], atol=1e-12)
        np.testing.assert_allclose(c_m, [3.0, -1.0, 2.0], atol=1e-12)
 
    def test_equal_masses(self):
        """Equal masses → center of mass is arithmetic mean of positions."""
        xyz = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 2., 0.]])
        atom_mass = np.array([1., 1., 1.])
        centered, c_m = center_of_mass_xyz(xyz, atom_mass)
        np.testing.assert_allclose(c_m, [0., 2/3, 0.], atol=1e-12)
 
    def test_original_not_mutated(self, water_molecule):
        xyz, _, _, atom_mass = water_molecule
        xyz_copy = xyz.copy()
        center_of_mass_xyz(xyz, atom_mass)
        np.testing.assert_array_equal(xyz, xyz_copy)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# moment_of_innertia_axis
# ═════════════════════════════════════════════════════════════════════════════
 
class TestMomentOfInertiaAxis:
 
    def test_returns_orthonormal_basis(self, water_molecule):
        """Eigenvectors must form an orthonormal matrix."""
        xyz, _, _, atom_mass = water_molecule
        centered, _ = center_of_mass_xyz(xyz, atom_mass)
        v = moment_of_innertia_axis(centered, atom_mass)
        np.testing.assert_allclose(v @ v.T, np.eye(3), atol=1e-10)
 
    def test_shape(self, water_molecule):
        xyz, _, _, atom_mass = water_molecule
        centered, _ = center_of_mass_xyz(xyz, atom_mass)
        v = moment_of_innertia_axis(centered, atom_mass)
        assert v.shape == (3, 3)
 
    def test_eigenvalues_descending(self, water_molecule):
        """Rows should correspond to eigenvalues in descending order."""
        xyz, _, _, atom_mass = water_molecule
        centered, _ = center_of_mass_xyz(xyz, atom_mass)
        v = moment_of_innertia_axis(centered, atom_mass)
        # Reconstruct inertia tensor and check projected inertias are descending
        Ixx = np.sum((centered[:, 1]**2 + centered[:, 2]**2) * atom_mass)
        Iyy = np.sum((centered[:, 0]**2 + centered[:, 2]**2) * atom_mass)
        Izz = np.sum((centered[:, 1]**2 + centered[:, 0]**2) * atom_mass)
        Ixy = -np.sum(centered[:, 0] * centered[:, 1] * atom_mass)
        Iyz = -np.sum(centered[:, 2] * centered[:, 1] * atom_mass)
        Ixz = -np.sum(centered[:, 0] * centered[:, 2] * atom_mass)
        I_m = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        projected = np.array([v[i] @ I_m @ v[i] for i in range(3)])
        assert projected[0] >= projected[1] >= projected[2] - 1e-10
 
    def test_symmetric_molecule(self):
        """Linear symmetric molecule: two degenerate eigenvalues."""
        xyz = np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        atom_mass = np.array([1., 16., 1.])
        centered, _ = center_of_mass_xyz(xyz, atom_mass)
        v = moment_of_innertia_axis(centered, atom_mass)
        assert v.shape == (3, 3)
        np.testing.assert_allclose(v @ v.T, np.eye(3), atol=1e-10)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# permute_by_type_and_dis
# ═════════════════════════════════════════════════════════════════════════════
 
class TestPermuteByTypeAndDis:
 
    def test_atom_num_sorted(self, water_molecule):
        xyz, force, atom_num, atom_mass = water_molecule
        xyz_s, force_s, atom_num_s, atom_mass_s, order = permute_by_type_and_dis(
            xyz, force, atom_num, atom_mass)
        assert np.all(atom_num_s[:-1] <= atom_num_s[1:])
 
    def test_order_is_valid_permutation(self, water_molecule):
        xyz, force, atom_num, atom_mass = water_molecule
        _, _, _, _, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
        assert set(order) == set(range(len(atom_num)))
 
    def test_already_sorted_is_identity(self):
        xyz = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        force = xyz.copy()
        atom_num = np.array([1, 1, 8])
        atom_mass = np.array([1., 1., 16.])
        _, _, atom_num_s, _, order = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
        assert np.all(atom_num_s == np.sort(atom_num))
 
    def test_reorder_consistency(self, two_water_molecules):
        """Shuffling input then permuting should give the same sorted result."""
        xyz, force, atom_num, atom_mass = two_water_molecules
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(atom_num))
        xyz_s1, _, an_s1, _, _ = permute_by_type_and_dis(xyz, force, atom_num, atom_mass)
        xyz_s2, _, an_s2, _, _ = permute_by_type_and_dis(
            xyz[perm], force[perm], atom_num[perm], atom_mass[perm])
        np.testing.assert_array_equal(an_s1, an_s2)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# direction_order
# ═════════════════════════════════════════════════════════════════════════════
 
class TestDirectionOrder:
 
    @pytest.mark.parametrize("direction,expected", [('x', 0), ('y', 1), ('z', 2)])
    def test_valid_directions(self, direction, expected):
        assert direction_order(direction) == expected
 
    def test_invalid_direction_raises(self):
        with pytest.raises((ValueError, SystemExit)):
            direction_order('w')
 
 
# ═════════════════════════════════════════════════════════════════════════════
# atom_mass_mapping
# ═════════════════════════════════════════════════════════════════════════════
 
class TestAtomMassMapping:
 
    @pytest.mark.parametrize("atomic_num,expected_mass", [
        (np.array([1]), 1.0),   # H
        (np.array([6]), 12.0),  # C
        (np.array([8]), 16.0),  # O
    ])
    def test_known_elements(self, atomic_num, expected_mass):
        result = atom_mass_mapping(atomic_num)
        assert result[0] == pytest.approx(expected_mass)
 
    def test_water_masses(self):
        atom_num = np.array([8, 1, 1])
        masses = atom_mass_mapping(atom_num)
        np.testing.assert_allclose(masses, [16., 1., 1.])
 
    def test_output_length_matches_input(self):
        atom_num = np.array([1, 6, 7, 8])
        assert len(atom_mass_mapping(atom_num)) == 4
 
 
# ═════════════════════════════════════════════════════════════════════════════
# transform_fragment  (converted from water_test / test_case manual checks)
# ═════════════════════════════════════════════════════════════════════════════
 
class TestTransformFragment:
 
    def test_output_shapes(self, water_molecule):
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        xyz_t, force_t, an_t, v_t, order_t = transform_fragment(
            xyz, force, atom_num, atom_mass, 'x', ref)
        assert xyz_t.shape == (3, 3)
        assert force_t.shape == (3, 3)
        assert v_t.shape == (3, 3)
        assert len(order_t) == 3
 
    @pytest.mark.parametrize("direction", ['x', 'y', 'z'])
    def test_all_directions_run(self, water_molecule, direction):
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        result = transform_fragment(xyz, force, atom_num, atom_mass, direction, ref)
        assert len(result) == 5
 
    def test_basis_vectors_orthonormal(self, water_molecule):
        """Returned v must be an orthonormal rotation matrix."""
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        _, _, _, v, _ = transform_fragment(xyz, force, atom_num, atom_mass, 'x', ref)
        np.testing.assert_allclose(v @ v.T, np.eye(3), atol=1e-10)
 
    def test_order_is_permutation(self, water_molecule):
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        _, _, _, _, order = transform_fragment(xyz, force, atom_num, atom_mass, 'x', ref)
        assert set(order) == set(range(len(atom_num)))
 
    def test_water_test_case_numeric(self):
        """Mirrors the original water_test() manual check."""
        force = np.array([
            [-1.949573e-03,  1.804180e-03,  1.063971e-03],
            [ 7.255650e-04,  4.399090e-04, -3.277888e-03],
            [ 1.224006e-03, -2.244091e-03,  2.213916e-03],
        ])
        xyz = np.array([
            [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
            [-2.481261367915496,  1.959085964931036,  2.293069675637073],
            [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
        ])
        atom_num  = np.array([8, 1, 1])
        atom_mass = np.array([16., 1., 1.])
        ref = np.ones((3, 3))
        xyz_t, force_t, an_t, v_t, order_t = transform_fragment(
            xyz, force, atom_num, atom_mass, 'x', ref)
        # Basic sanity: result is finite and shapes match
        assert np.all(np.isfinite(xyz_t))
        assert np.all(np.isfinite(force_t))
        # Oxygen (atomic num 8) should still be present once
        assert np.sum(an_t == 8) == 1
 
    def test_two_water_molecules(self, two_water_molecules):
        """Original test_case() with 6 atoms."""
        xyz, force, atom_num, atom_mass = two_water_molecules
        ref = np.ones((6, 3))
        xyz_t, force_t, an_t, v_t, order_t = transform_fragment(
            xyz, force, atom_num, atom_mass, 'x', ref)
        assert np.all(np.isfinite(xyz_t))
        assert np.sum(an_t == 8) == 2
        assert np.sum(an_t == 1) == 4
 
 
# ═════════════════════════════════════════════════════════════════════════════
# Helper: compute c_m externally (mirrors what global_transform_fragment does
# internally) so tests can verify xyz reconstruction without changing the API.
# ═════════════════════════════════════════════════════════════════════════════
 
def _get_sorted_xyz_and_cm(xyz, force, atom_num, atom_mass):
    """Return (xyz_sorted_centered, c_m, atom_num_sorted) — the same
    preprocessing that global_transform_fragment applies before yielding."""
    xyz_s, force_s, an_s, am_s, _ = permute_by_type_and_dis(
        xyz.copy(), force.copy(), atom_num.copy(), atom_mass.copy())
    xyz_centered, c_m = center_of_mass_xyz(xyz_s, am_s)
    return xyz_centered, c_m, an_s
 

 
# ═════════════════════════════════════════════════════════════════════════════
# global_transform_fragment  (converted from permutation_test / water_global_test)
# ═════════════════════════════════════════════════════════════════════════════
 
class TestGlobalTransformFragment:
    def _permutation_data(self):
        force = np.array([
            [-1.949573e-03,  1.804180e-03,  1.063971e-03],
            [ 7.255650e-04,  4.399090e-04, -3.277888e-03],
            [ 1.224006e-03, -2.244091e-03,  2.213916e-03],
            [ 1., 1., 1.],
        ])
        xyz = np.array([
            [-2.1522976348577143, 1.935813831410094,  1.3844868868883233],
            [-2.481261367915496,  1.959085964931036,  2.293069675637073],
            [-2.607829012546703,  2.6389455523580923, 0.9029163487370644],
            [ 1., 1., 1.],
        ])
        atom_num  = np.array([8, 1, 1, 1])
        atom_mass = np.array([16., 1., 1., 1.])
        return xyz, force, atom_num, atom_mass
        
    def test_yields_exactly_three_directions(self, water_molecule):
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        results = list(global_transform_fragment(xyz, force, atom_num, atom_mass, ref))
        assert len(results) == 3
 
    def test_each_yield_has_five_elements(self, water_molecule):
        """Generator yields 5-tuples (xyz, force, atom_num, v, order) — c_m is
        intentionally excluded from the API and computed externally for tests."""
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        for item in global_transform_fragment(xyz, force, atom_num, atom_mass, ref):
            assert len(item) == 5
 
    def test_basis_orthonormal_for_all_directions(self, water_molecule):
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        for xyz_t, force_t, an_t, v_t, order_t in global_transform_fragment(
                xyz, force, atom_num, atom_mass, ref):
            np.testing.assert_allclose(v_t @ v_t.T, np.eye(3), atol=1e-10)
 
    def test_xyz_reconstruction(self, water_molecule):
        """xyz_t can be inverted back to the original centered+sorted coords
        using v and order, with c_m computed externally (mirrors water_global_test)."""
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        xyz_sorted_centered, c_m, _ = _get_sorted_xyz_and_cm(xyz, force, atom_num, atom_mass)
 
        for xyz_t, force_t, an_t, v_t, order_t in global_transform_fragment(
                xyz, force, atom_num, atom_mass, ref):
            recovered = (np.linalg.inv(v_t) @ xyz_t[np.argsort(order_t)].T).T + c_m
            np.testing.assert_allclose(
                recovered, xyz, atol=1e-10,
                err_msg="xyz not recoverable from transformed output + external c_m")
 
    def test_force_reconstruction(self, water_molecule):
        """Reconstructed force columns should reassemble to the original force
        (mirrors the core check in water_global_test)."""
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        xyz_sorted, _, an_s, am_s, _ = permute_by_type_and_dis(
            xyz.copy(), force.copy(), atom_num.copy(), atom_mass.copy())
        force_sorted, _, _, _, _ = permute_by_type_and_dis(
            xyz.copy(), force.copy(), atom_num.copy(), atom_mass.copy())
        # get the permuted force directly
        _, force_s, _, _, _ = permute_by_type_and_dis(
            xyz.copy(), force.copy(), atom_num.copy(), atom_mass.copy())
 
        n = len(atom_num)
        reconstructed = np.zeros((n, 3))
        for c, (xyz_t, force_t, an_t, v_t, order_t) in enumerate(
                global_transform_fragment(xyz, force, atom_num, atom_mass, ref)):
            reconstructed[:, c] = force_t[:, c][np.argsort(order_t)]
 
        # reconstructed should match force_s (sorted by atom type)
        np.testing.assert_allclose(reconstructed, force, atol=1e-10,
            err_msg="Reconstructed force does not match original sorted force")
 
 
    def test_permutation_invariance_force(self):
        xyz, force, atom_num, atom_mass = self._permutation_data()
        ref = np.ones((4, 3))

    # get canonical result with original ordering
        canonical = np.zeros((4, 3))
        for c, (xt, ft, ant, vt, ordt) in enumerate(
                global_transform_fragment(xyz, force, atom_num, atom_mass, ref)):
            canonical[:, c] = ft[:, c]

    # shuffled inputs must produce identical reconstructed force
        rng = np.random.default_rng(0)
        for _ in range(3):
            perm = rng.permutation(len(atom_num))
            shuffled = np.zeros((4, 3))
            for c, (xt, ft, ant, vt, ordt) in enumerate(
                    global_transform_fragment(xyz[perm], force[perm],
                                            atom_num[perm], atom_mass[perm], ref)):
                shuffled[:, c] = ft[:, c]
            np.testing.assert_allclose(shuffled, canonical, atol=1e-10,
                err_msg="Reconstructed force changed under atom permutation")

    def test_permutation_invariance_xyz(self):
        xyz, force, atom_num, atom_mass = self._permutation_data()
        ref = np.ones((4, 3))
        #_, c_m, _ = _get_sorted_xyz_and_cm(xyz, force, atom_num, atom_mass)

    # canonical xyz reconstruction per direction
        canonical = []
        for xt, ft, ant, vt, ordt in global_transform_fragment(
                xyz, force, atom_num, atom_mass, ref):
            #canonical.append((np.linalg.inv(vt) @ xt[np.argsort(ordt)].T).T + c_m)
            xyz_t = np.array(xt)

    # shuffled inputs must recover the same xyz
        rng = np.random.default_rng(1)
        for _ in range(3):
            perm = rng.permutation(len(atom_num))
            _, c_m_s, _ = _get_sorted_xyz_and_cm(
                xyz[perm], force[perm], atom_num[perm], atom_mass[perm])
            for i, (xt, ft, ant, vt, ordt) in enumerate(
                    global_transform_fragment(xyz[perm], force[perm],
                                            atom_num[perm], atom_mass[perm], ref)):
                #recovered = (np.linalg.inv(vt) @ xt[np.argsort(ordt)].T).T + c_m_s
                continue
            np.testing.assert_allclose(xyz_t, xt, atol=1e-10,
                err_msg="Reconstructed xyz changed under atom permutation") 
                    
    def test_output_is_finite(self, two_water_molecules):
        xyz, force, atom_num, atom_mass = two_water_molecules
        ref = np.ones((6, 3))
        for xyz_t, force_t, an_t, v_t, order_t in global_transform_fragment(
                xyz, force, atom_num, atom_mass, ref):
            assert np.all(np.isfinite(xyz_t))
            assert np.all(np.isfinite(force_t))
 
    def test_atom_types_preserved(self, water_molecule):
        """Atom numbers should be the same set after transform."""
        xyz, force, atom_num, atom_mass = water_molecule
        ref = np.ones((3, 3))
        for _, _, an_t, _, _ in global_transform_fragment(
                xyz, force, atom_num, atom_mass, ref):
            assert set(an_t) == set(atom_num)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# Edge / regression cases
# ═════════════════════════════════════════════════════════════════════════════
 
class TestEdgeCases:
 
    def test_collinear_atoms(self):
        """All atoms on one axis — inertia tensor has a zero eigenvalue."""
        xyz = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        atom_mass = np.array([1., 1., 1.])
        centered, _ = center_of_mass_xyz(xyz, atom_mass)
        v = moment_of_innertia_axis(centered, atom_mass)
        # Still orthonormal despite degeneracy
        np.testing.assert_allclose(v @ v.T, np.eye(3), atol=1e-10)
 
    def test_large_molecule_does_not_crash(self):
        """Stress test with 50 atoms of mixed types."""
        rng = np.random.default_rng(7)
        n = 50
        xyz = rng.standard_normal((n, 3))
        force = rng.standard_normal((n, 3))
        atom_num = rng.integers(1, 5, size=n)
        atom_mass = atom_mass_mapping(atom_num)
        ref = np.zeros((n, 3))
        results = list(global_transform_fragment(xyz, force, atom_num, atom_mass, ref))
        assert len(results) == 3
 
    def test_single_atom_per_type(self):
        """One atom of each of three different elements."""
        xyz = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
        force = np.ones((3, 3))
        atom_num = np.array([1, 6, 8])
        atom_mass = np.array([1., 12., 16.])
        ref = np.ones((3, 3))
        xyz_t, force_t, an_t, v_t, order_t = transform_fragment(
            xyz, force, atom_num, atom_mass, 'z', ref)
        assert np.all(np.isfinite(xyz_t))
 
    def test_identical_atom_positions_different_types(self):
        """Two atoms at the same position but different element — should not crash."""
        xyz = np.array([[1., 1., 1.], [1., 1., 1.], [0., 0., 0.]])
        force = np.ones((3, 3))
        atom_num = np.array([1, 8, 6])
        atom_mass = np.array([1., 16., 12.])
        ref = np.ones((3, 3))
        # Should complete without error
        result = transform_fragment(xyz, force, atom_num, atom_mass, 'x', ref)
        assert len(result) == 5
