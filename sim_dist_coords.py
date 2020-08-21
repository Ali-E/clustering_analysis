"""This file contais the functions related to converting different
formats of the input data to one another, e.g. similarity graph to
coordinates.
"""
import numpy as np
from sklearn.manifold import *
from sklearn import preprocessing
import sympy


def get_all_ids(sim_dict):
    """Generates sorted list of ids from a dictionary with id tuples as keys.

    Args:
        sim_dict: a similarity dictionary in which the keys are tuples of
        pair of ids.

    Returns:
        A sorted list of the ids contained in the tuples (keys of the
        dictionary).
    """
    all_ids = set()
    for id_tup in sim_dict.keys():
        all_ids.add(id_tup[0])
        all_ids.add(id_tup[1])
    return sorted(list(all_ids))


def lists_to_dict(lst1, lst2):
    """Given two lists generates a dict with keys and values from the lists.

    Args:
        lst1: a list from which the keys of the dictionary are formed.
        lst2: a list from which the values of the dictionary are formed.

    Returns:
        A dictionary in which the key of each value is taken from lst1. the 
        value in the same index as the key from lst2 is considered the value
        of the key. If an element is repeated in lst1, the value
        for the last one is considered for that key.
    """
    lst1_to_lst2_list = zip(lst1, lst2)
    lst1_to_lst2_dict = dict(lst1_to_lst2_list)
    return lst1_to_lst2_dict


def dict_to_mat(sim_dict, data_size=None, max_val=1.0):
    """Generates a matrix of the values from a given similarity dictionary.
    
    Args:
        sim_mat: A similarity dictionary in which the keys are tuple of pair of
        ids and values are the similarity values.

    Returns:
        sim_mat: A matrix which contains the similairy values.
        id_to_idx_dict: A mapping of id (used in the dictionary)
            to the corresponding row index of the matrix.
        idx_to_id_dict: A mapping of row index to the original id
            used in the dictionary.
    """
    
    if data_size is None:
        all_ids = get_all_ids(sim_dict)  #Extracts all the ids from the keys that are tuples of ids
        data_size = len(all_ids)
    else:
        all_ids = list(range(data_size))

    idx_list = range(data_size)
    sim_mat = np.zeros((data_size, data_size))
    sim_mat = sim_mat + (np.eye(sim_mat.shape[0]) * max_val)  #For self-similarities the max value is used

    id_to_idx_dict = lists_to_dict(all_ids, idx_list)
    idx_to_id_dict = lists_to_dict(idx_list, all_ids)

    for i in range(data_size):
        for j in range(i):
            i_id = idx_to_id_dict[i]
            j_id = idx_to_id_dict[j]

            i_j_value = 0.0
            if (i_id, j_id) in sim_dict:
                i_j_value = sim_dict[(i_id, j_id)]
            elif (j_id, i_id) in sim_dict:
                i_j_value = sim_dict[(j_id, i_id)]
            # else:
            #     raise KeyError

            sim_mat[i, j] = i_j_value
            sim_mat[j, i] = i_j_value

    return sim_mat, id_to_idx_dict, idx_to_id_dict


def normalize(mat):
    """normalizes a numpy nd-array.
    
    Args:
        mat: numpy nd-array

    Returns:
        The normalized nd-array where the values range from 0.0 to 1.0.
    """
    normalized_mat = (mat - mat.min()) / float(mat.max() - mat.min())
    return normalized_mat


def sim_dist_convert(dist_mat, normalize_flag=False):
    """Converts a similarity matrix to a distance one and vice versa.

    Args:
        sim_mat: An nd-array of values.
        normalize_flag: If set to true, normalizes the input matrix to have
            a range of 0 to 1 (default: True).

    Returns:
        dist_mat: An nd-array in which the values are computed by decreasing 
        the values of the input matrix from its maximum value.
    """
    if normalize_flag:
        normalized_dist_mat = normalize(dist_mat)
        sim_mat = normalized_dist_mat.max() - normalized_dist_mat
    else:
        sim_mat = dist_mat.max() - dist_mat
    return sim_mat


def compute_dist_mat(coords):
    """Generates a distance matrix given a list of coordinates of points.

    Args:
        coords: A list in which each element is another list containing the
        coordinate of a point in a Euclidean space.

    Returns:
        A matrix in which the value of the elemeng i,j is the 
        Euclidean distance of the point i and j in the coords list.
    """
    data_size = len(coords)
    dist_mat = np.zeros((data_size, data_size))
    for i in range(data_size):
        for j in range(i):
            dist_mat[i, j] = np.linalg.norm(coords[i, :]-coords[j, :])
            dist_mat[j, i] = dist_mat[i, j]

    return np.array(dist_mat).astype(float)


def compute_dot_mat(coords, normalize_flag=True):
    if normalize_flag:
        normalized_coords = preprocessing.normalize(coords, norm='l2')
        return np.round(np.matmul(normalized_coords, normalized_coords.T), 5)
    return np.round(np.matmul(coords, np.array(coords).T), 5)


def mat_to_dict(sim_mat, no_zeros=False):
    """Converts a similarity matrix to a similarity dictionary.

    Args:
        sim_mat: A similarity matrix (assumed to be symmetric).
    
    Returns:
        sim_dict: A dictionary that will contain tuple of row ids as the keys
        and the similarity values as the values. The generated dictionary
        will not include the identity mappings (assumed to have a value of 1).
    """
    sim_dict = {}
    for i in range(len(sim_mat)):
        for j in range(i):
            if no_zeros and float(sim_mat[i, j]) == 0.00:
                continue
            sim_dict[(i, j)] = float(sim_mat[i, j])
    return sim_dict


def get_coords_equation(dist_mat):
    """Given a distance matrix finds set of points that have those distances.

    This function assums points exist in a space of which the dimention is
    one less than the number of points. Then consideres the first point as 
    the center of the space. In an iterative fasion adds each new point 
    such that it preserves the required distance to the earlier added points.
    Args:
        dist_mat: A matrix (2d array) of distance between pair of points.
    
    Returns:
        A numpy array containing the coordinate of the points.
    """
    data_size = len(dist_mat)
    coords = sympy.symarray('x', (data_size, data_size-1)) # generating a 2d array of variables
    coords = np.tril(coords, k=-1) # considering only the lower triangle

    for new_point_idx in range(1, data_size):
        print('>', new_point_idx)
        all_equations = []

        for solved_point_idx in range(new_point_idx):
            equation_exp = np.sum((coords[solved_point_idx] - coords[new_point_idx])**2)
            equation_val = dist_mat[solved_point_idx, new_point_idx]**2
            equation = equation_exp - equation_val
            all_equations.append(equation)

        solution = sympy.solve(all_equations, list(coords[new_point_idx, :new_point_idx]))[1]
        coords[new_point_idx, :new_point_idx] = solution

    return np.array(coords).astype(float)


def get_coords_mds(dist_mat, output_dim=None, hyper_params={}):
    """Given a distance matrix return a set of points with those distance using MDS.

    This function uses Multidimentional Scaling (MDS) method to find
    a set of points with those distances. The complexity of this function
    is equal to the matrix multiplication because of the eigendecompoition step.
    Args:
        dist_mat: A matrix (2d array) of distance between pair of points.
        precision_boost: Due to the limitation in the precision points and 
            errors caused by rounding to overcome this limitation, the computed
            eigenvalues might change to small negative values which causes 
            problem when computung the sqrt of them. Therefore,
            a small factor of the identity matrix is added to matrix M before
            its eigendecomposition. The value of this parameter defines the 
            constant factor that is muliplied by the identity matrix. If its
            value is set to 'AUTO', then the value of the precision_boost will
            set so that the lowest value of the diagonal of the M matrix will
            become 0 plus epsilon (0.0000001).

    Returns:
        A numpy array containing the coordinate of the points.
    """
    params = {'precision_boost': 0.000001}
    params.update(hyper_params)

    precision_boost = params['precision_boost']
    top_eigenvals = output_dim

    data_size = len(dist_mat)
    M = np.zeros((data_size, data_size))

    min_diag_value=10000.0
    for i in range(data_size):
        for j in range(i+1):
            M[i, j] = (dist_mat[0, j]**2 + dist_mat[i, 0]**2 - dist_mat[i, j]**2)/2.0
            M[j, i] = M[i, j]

            if i == j and M[i, j] < min_diag_value:
                min_diag_value = M[i, j]

    if str(precision_boost) == 'AUTO':
        precision_boost = max(0.0000001, 0.0000001-min_diag_value)
    M += precision_boost * np.eye(M.shape[0])

    # try:
    #     np.linalg.cholesky(M)
    # except Exception as e:
    #     print(e)

    eigenvals, eigenvects = np.linalg.eigh(M)
    if np.any(eigenvals < 0):
        print('Matrix is not positive semi-definite')
        print(eigenvals)

    if top_eigenvals is not None:
        orders = np.array(range(len(eigenvals)))
        combined = list(zip(orders, eigenvals))
        sorted_arr = sorted(combined, key=lambda x: x[1], reverse=True)
        eigenvals = [elem[1] for elem in sorted_arr[:top_eigenvals]]
        eigenvects = eigenvects[:, np.array([elem[0] for elem in sorted_arr])]
        eigenvects = eigenvects[:, :top_eigenvals]

    eigenvals_diag_mat = np.diag(np.sqrt(eigenvals))
    
    coords = np.matmul(eigenvects, eigenvals_diag_mat)
    coords = coords[:, ~np.all(coords==0, axis=0)]
    return coords


def get_coords_mds_stress(dist_mat, output_dim=2, hyper_params={}):
    """This function uses the metric MDS with stress loss function to recover the coordinates.

    Args:
        dist_mat: The matrix containing the pair-wise distances.
        output_dim: The dimension of the space we want to map the points into.
        hyper_params: The hyperparameters of the function to be called:

    Returns:
        coords: A matrix of size (n, output_dim) that contains the coordinate of
            the points in a space with the desired dimension.
    """
    params = {'max_iter': 1000,
              'dissimilarity': 'precomputed',
              'eps': 0.000001}
    params.update(hyper_params)

    embedding = MDS(max_iter=params['max_iter'],
                    n_components=output_dim,
                    dissimilarity=params['dissimilarity'],
                    eps=params['eps'])
    coords = embedding.fit_transform(dist_mat)
    return coords


def get_coords_tsne(dist_mat, output_dim=2, hyper_params={}):
    """This function uses t-SNE embed the points in a 2d or 3d space. 

    Args:
        dist_mat: The matrix containing the pair-wise distances.
        output_dim: The dimension of the space we want to map the points into.
            This can be 2 or 3 for t-SNE according to the documentation of sklearn.
        hyper_params: The hyperparameters of the function to be called:

    Returns:
        coords: A matrix of size (n, output_dim) that contains the coordinate of
            the points in a space with the desired dimension.
    """
    params = {'metric': 'precomputed'}
    params.update(hyper_params)

    embedding = TSNE(n_components=output_dim,
                     metric=params['metric'])
    coords = embedding.fit_transform(dist_mat)
    return coords


def get_coords_isomap(dist_mat, output_dim=2, hyper_params={}):
    """This function uses Isomap to recover the coordinates.

    Args:
        dist_mat: The matrix containing the pair-wise distances.
        output_dim: The dimension of the space we want to map the points into.
        hyper_params: The hyperparameters of the function to be called:

    Returns:
        coords: A matrix of size (n, output_dim) that contains the coordinate of
            the points in a space with the desired dimension.
    """
    params = {'n_neighbors': 100,
              'metric': 'precomputed'}
    params.update(hyper_params)

    embedding = Isomap(n_neighbors=params['n_neighbors'],
                       n_components=output_dim,
                       metric=params['metric'])
    coords = embedding.fit_transform(dist_mat)
    return coords


def sim_dict_to_coords(sim_dict, embedding_method=get_coords_mds, data_size=None, return_sim_mat=False, top_eigenvals=None, hyper_params={}):
    """Finds a set of points with the distance corresponding to a similarity dictionary.

    Given a similarity dictionary, it first finds the corresponding similarity matrix,
    and then forms the corresponding distance matrix. Finally, find a set of points
    with those distances.
    Args:
        sim_mat: A similarity dictionary in which the keys are tuple of pair of
        ids and values are the similarity values.
        return_sim_mat: If True, the similarity matrix is also returned.

    Returns:
        coords: A numpy array containing the coordinate of the points.
        sim_mat (if return_sim_mat is set to True): The similarity matrix
        id_to_idx_dict: A mapping of id (used in the dictionary)
            to the corresponding row index of the matrix.
        idx_to_id_dict: A mapping of row index to the original id
            used in the dictionary.
    """
    sim_mat, id_to_idx_dict, idx_to_id_dict = dict_to_mat(sim_dict, data_size)
    dist_mat = sim_dist_convert(sim_mat, normalize_flag=True)
    coords = embedding_method(dist_mat, output_dim=top_eigenvals, hyper_params=hyper_params)

    if return_sim_mat:
        return coords, sim_mat, id_to_idx_dict, idx_to_id_dict
    return coords, id_to_idx_dict, idx_to_id_dict
