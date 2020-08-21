import numpy as np
import sim_dist_coords as sdc
import unittest


def almost_equal(arr1, arr2, threshold=0.00001):
    arr1 = np.array(arr1)
    if np.shape(arr1) != np.shape(arr2):
        return False
    
    arr1_flat = arr1.reshape(arr1.size)
    arr2_flat = arr2.reshape(arr2.size)

    for idx in range(len(arr1_flat)):
        if abs(arr1_flat[idx] - arr2_flat[idx]) > threshold:
            return False
    return True


class TestSimDistCoords(unittest.TestCase):

    def test_get_all_ids(self):
        sample_dict = {('a0', 'a1'): 1, ('a1', 'b1'): 2, ('b2', 'b1'): 3}
        self.assertEqual(sdc.get_all_ids(sample_dict),
                        ['a0', 'a1', 'b1', 'b2'])


    def test_lists_to_dict(self):
        lst1 = ['a0', 'b0', 'b1', 'b2']
        lst2 = range(4)
        self.assertEqual(sdc.lists_to_dict(lst1, lst2),
                        {'a0': 0, 'b0': 1, 'b1': 2, 'b2': 3})
        self.assertEqual(sdc.lists_to_dict(lst2, lst1),
                        {0: 'a0', 1: 'b0', 2: 'b1', 3: 'b2'})


    def test_dict_to_mat(self):
        sim_dict = {('a1', 'b1'): 1, ('a0', 'a1'): 2, ('b1', 'a0'): 3}
        self.assertTrue(len(sdc.dict_to_mat(sim_dict)) == 3)
    
        sim_mat = [[4, 2, 3], [2, 4, 1], [3, 1, 4]]
        id_to_idx_dict = {'a0': 0, 'a1': 1, 'b1': 2}
        idx_to_id_dict = {0: 'a0', 1: 'a1', 2: 'b1'}
        
        given_answer = sdc.dict_to_mat(sim_dict, max_val=4)
        self.assertTrue(np.all(given_answer[0] == sim_mat))
        self.assertEqual(given_answer[1], id_to_idx_dict)
        self.assertEqual(given_answer[2], idx_to_id_dict)


    def test_dict_to_mat_exception(self):
        sim_dict = {('a0', 'a1'): 2, ('b1', 'a0'): 3}
        with self.assertRaises(KeyError):
            sdc.dict_to_mat(sim_dict)
        
        sim_dict = {('b0', 'a1'): 5, ('b1', 'b0'): 4, 
                ('a1', 'b1'): 1, ('a0', 'a1'): 2, ('b1', 'a0'): 3}
        with self.assertRaises(KeyError):
            sdc.dict_to_mat(sim_dict)


    def test_sim_dist_convert(self):
        sim_mat = np.array([[5, 2, 3], [2, 5, 1], [3, 1, 5]])
        dist_mat = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])

        self.assertTrue(np.all(sdc.sim_dist_convert(sim_mat) == dist_mat))

    
    def test_get_coords_equation(self):
        b = np.sqrt(2)
        dist_mat = np.array([[0, 1, 1, 1],
                             [1, 0, b, b],
                             [1, b, 0, b],
                             [1, b, b, 0]])

        coords = sdc.get_coords_equation(dist_mat)
        dist_mat_new = sdc.compute_dist_mat(coords)
        self.assertTrue(almost_equal(dist_mat_new, dist_mat))


    def test_get_coords_mds(self):
        b = np.sqrt(2)
        dist_mat = np.array([[0, 1, 1, 1],
                             [1, 0, b, b],
                             [1, b, 0, b],
                             [1, b, b, 0]])

        coords = sdc.get_coords_mds(dist_mat)
        dist_mat_new = sdc.compute_dist_mat(coords)
        self.assertTrue(almost_equal(dist_mat_new, dist_mat))


if __name__ == '__main__':
    unittest.main()
