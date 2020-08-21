"""This module contains function for creating various
sample datasets for testing clustering methods.
"""
import datetime
from hmmlearn import hmm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import powerlaw
from sklearn import datasets
import sim_dist_coords as sdc 
import time

from main import make_id_cluster_maps


def generate_data(N=10, seed=4711):
    """Generates guassian distributions each of which having a center on a point of a grid.
    Args:
        N: the size of one side of the grid.
    """
    np.random.seed(seed)  # for repeatability
    centers = []
    for i in range(0, 2*N, 2):
        for j in range(0, 2*N, 2):
            centers.append([i, j])
    n = 100
    X = np.zeros((0, 2))
    for idx, c in enumerate(centers):
        points = np.random.multivariate_normal(c, [[0.4 ** 2, 0], [0, 0.4 ** 2]], size=[30,]) 
        X = np.concatenate((X, points),)
    return X


def noisy_circle(hyper_params={}):
    """Generates two concentric circles each wich gaussin noise added.

    Args:
        hyper_params: a dictionary that contains the value of the hyper-parameters
        of the function. (explained below)
    """
    params = {'n_samples': 1500, # number of points (can be a tuple with the number for each circle)
              'factor': 0.6, # scale factor between the two circles (0< factor <1)
              'noise': 0.1, # this is the noise added to each circle)
              'random_state': 8}
    params.update(hyper_params)
    X, y = datasets.make_circles(n_samples=params['n_samples'],
                                 factor=params['factor'],
                                 noise=params['noise'],
                                 random_state=params['random_state'])
    return X, y


def noisy_moons(hyper_params={}):
    """Generates two half circles each with noise added.

    Args:
        hyper_params: a dictionary that contains the value of the hyper-parameters
        of the function. (explained below)
    """
    params = {'n_samples': 1500, # number of points (can be a tuple with the number for each circle)
              'noise': 0.05, # the noise added to each half-circle
              'random_state': 8}
    params.update(hyper_params)
    X, y = datasets.make_moons(n_samples=params['n_samples'],
                               noise=params['noise'])
    return X, y


def blobs(hyper_params={}):
    """Generates gaussian distributions for which the center are chosen randomly.

    Args:
        hyper_params: a dictionary of the parameters (explained below).
    """
    params = {'n_samples': 1500, # number of points (can be a list of numbers fo each blob)
              'n_features': 2, # dimention of the points
              'centers': 3, # number of centers (if n_samples is a list it should be None or the same number)
              'cluster_std': 1.0, # std of the blobls (can be a list of numbers)
              'center_box': (-10.0, 10.0), # the bounding box for each blob
              'transformation': None, # transformation of data, e.g. [[0.6, -0.6], [-0.4, 0.8]]
              'random_state': 8}
    params.update(hyper_params)
    X, y = datasets.make_blobs(n_samples=params['n_samples'],
                               n_features=params['n_features'],
                               centers=params['centers'],
                               cluster_std=params['cluster_std'],
                               center_box=params['center_box'],
                               random_state=params['random_state'])

    if params['transformation'] != None:
        X = np.dot(X, params['transformation'])

    return X, y


def synthetic_CC(hyper_params={}):
    """Generates synthetic CCs by gaussian distributions for which the center are chosen randomly.

    Args:
        hyper_params: a dictionary of the parameters (explained below).
    """
    params = {'n_samples': 150, # number of points (can be a list of numbers fo each blob)
              'n_features': 2, # number of points (can be a list of numbers fo each blob)
              'centers': 7, # number of centers (if n_samples is a list it should be None or the same number)
              'cluster_std': 1.0, # std of the blobls (can be a list of numbers)
              'std_min': 0.0, # minimum std considered for each cluster
              'std_max': 1.0, # maximum std considered for each cluster
              'diff_val': 2, # the number of elements in each cluster will be average -+ this value
              'center_box': (-10.0, 10.0), # the bounding box for each blob
              'transformation': None, # transformation of data, e.g. [[0.6, -0.6], [-0.4, 0.8]]
              'random_state': 170}
    params.update(hyper_params)

    avg_number = int(params['n_samples']/params['centers'])

    points_number_list = []
    std_list = []
    diff_val = params['diff_val']
    for blob_idx in range(params['centers']):
        points_number = random.choice(range(avg_number-diff_val, avg_number+diff_val+1, 1))
        points_number_list.append(points_number)
        std = random.uniform(params['std_min'], params['std_max'])
        std_list.append(std)

    remaining_points = params['n_samples'] - np.sum(points_number_list)
    if remaining_points > 0:
        points_number_list.append(remaining_points)
        std = random.uniform(params['std_min'], params['std_max'])
        std_list.append(std)

    params['n_samples'] = points_number_list
    params['cluster_std'] = std_list
    params['centers'] = None

    X, y = blobs(params)
    return X, y


def generate_covar_mat(min_covar, max_covar):
    """Generates a 2x2 random covariance matrix.

    Args:
        min_covar: This value is assumed to be the minimum value of variance on each dimension. 
        max_covar: This value is assumed to be the maximum value of variance on each dimension.

    Returns:
        A 2x2 covariance matrix with variance values uniformly random chosen between min_covar
        and max_covar. The other values (which is the same for the 2 remianing values of the 
        matrix, is chosen such that the generated matrix is positive definite. 
    """
    a = np.random.uniform(min_covar, max_covar)
    d = np.random.uniform(min_covar, max_covar)
    b = np.random.uniform(0.0, np.sqrt(a*d))
    c = b
    return [[a, b], [c, d]]


def generate_time_stamps(hyper_params={}):
    """
    Generates time stamps for using power law to sample the average number of points per day,
    and using a poisson distribution to sample the the number of points per day.
    """
    params = {'period':30,
              'a': 0.2,
              'max_average':15,
              # 'start_date': np.datetime64('2020-07-18'),
              'start_date': datetime.datetime(2020,7,18,0,0,0,0),
              'random_state': 1234}
    params.update(hyper_params)
    
    np.random.seed(params['random_state'])
    max_avg = params['max_average']
    period = params['period']
    start_date = params['start_date']

    avg_num_per_day = int(np.ceil(powerlaw.rvs(params['a'], size=1)[0] * max_avg))
    num_per_day_lst = np.random.poisson(avg_num_per_day, period)

    # time_stamps = []
    # for idx, num in enumerate(num_per_day_lst):
    #     time_stamps += num*[start_date + idx]

    time_stamps = []
    for idx, num in enumerate(num_per_day_lst):
        new_times = []
        extra_seconds = (np.random.random(num)*3600*24).astype(int)
        extra_microseconds = (np.random.random(num)*1000000).astype(int)
        for i in range(num):
            diff_time = datetime.timedelta(idx, int(extra_seconds[i]), int(extra_microseconds[i]))
            new_time = start_date + diff_time
            unix_time = new_time.timestamp()
            new_times.append(unix_time)
        time_stamps += sorted(new_times)

    return time_stamps


def synthetic_CC_hmm(hyper_params={}):
    """Generates incremental synthetic data using hmm assuming gaussian dists for CCs.
    
    Args:
        hyper_params: A dictionary that could carry the new values for keys that are 
        already set to default values in the body of the function (params). These
        keys are:

            n_samples: The total number of points generates (corresponds to the number
                of sessions.
            CC_count: The total number of clusters (CCs)
            mean_min: Minimum value for the x and y coordinates of the center points of 
                the gaussian distributions.
            mean_max: Maximum value for the x and y coordinates of the center points.
            covar_min: Minimum value of the variance on each axis for each of the 
                gaussian distributions.
            covar_max: Maximum value of the variance on each axis.
            stationary_prob: The probability that we remain in the same CC on the next step.
                The transition probability to each other CC is set to be equal.
            random_state: The random seed for numpy.

    Returns:
        X: A list of size n_samples, each element representing a point in 2d.
        y: Contains the corresponding states of points in X.
    """
    params = {'n_samples': 90,
              'period': None,
              'CC_count': 11,
              'mean_min': 0.0,
              'mean_max': 1.0,
              'covar_min': 0.001,
              'covar_max': 0.01,
              'stationary_prob': 0.5,
              'random_state': 1234,
              'time_stamp_params': {}}
    params.update(hyper_params)

    time_stamp_params = params['time_stamp_params']
    time_stamp_params.update({'random_state': params['random_state']})
    
    np.random.seed(params['random_state'])
    period = params['period']
    if period is not None:
        time_stamps_lst = generate_time_stamps(time_stamp_params)
        n_samples = len(time_stamps_lst)
    else:
        n_samples = params['n_samples']

    CC_count = params['CC_count']
    stationary_prob = params['stationary_prob']

    if 'start_probs' not in params:
        start_probs = np.array([1.0/CC_count for i in range(CC_count)])
    else:
        start_probs = params['start_probs']

    if 'means' not in params:
        means = np.random.uniform(params['mean_min'], params['mean_max'], (CC_count, 2))
    else:
        means = params['means']

    if 'trans_mat' not in params:
        trans_mat = np.zeros((CC_count, CC_count), dtype=float)
        for row in range(CC_count):
            for col in range(CC_count):
                if row == col:
                    trans_mat[row, col] = stationary_prob
                else:
                    trans_mat[row, col] = (1.0-stationary_prob)/(CC_count-1)
    else:
        trans_mat = params['trans_mat']

    if 'covars' not in params:
        covars = []
        for i in range(CC_count):
            covars_temp = generate_covar_mat(params['covar_min'], params['covar_max'])
            covars.append(covars_temp)
        covars = np.array(covars)
    else:
        covars = params['covars']

    model = hmm.GaussianHMM(n_components=CC_count,
                            covariance_type="full",
                            init_params="cm",
                            params="cmt")

    model.startprob_ = start_probs
    model.transmat_ = trans_mat
    model.means_ = means
    model.covars_ = covars
    X, y = model.sample(n_samples)

    X_list = [[float(elem[0]), float(elem[1])] for elem in X]
    y_list = [int(label) for label in y]

    if period is None:
        return X_list, y_list, None
    else:
        # time_stamps_lst = [str(elem) for elem in time_stamps_lst]
        return X_list, y_list, time_stamps_lst


def make_synthetic_data(hyper_params={}):
    params = {'entitys_count': 10,
              'min_samples': 25,
              'max_samples': 500,
              'period': 30,
              'min_CCs': 4,
              'max_CCs': 30,
              'random_seed': 1234
              }
    params.update(hyper_params)
    
    np.random.seed(params['random_seed'])
    period = params['period']
    entitys_count = params['entitys_count']
    min_samples = params['min_samples']
    max_samples = params['max_samples']
    min_CCs = params['min_CCs']
    max_CCs = params['max_CCs']

    whole_data = {}
    for entity_idx in range(entitys_count):
        print('generated: ', entity_idx)
        n_samples = int(np.random.uniform(min_samples, max_samples))
        CC_count = int(np.random.uniform(min_CCs, max_CCs))

        single_params = {'n_samples': n_samples,
                      'period': period,
                      'CC_count': CC_count,
                      'mean_min': 0.0,
                      'mean_max': 1.0,
                      'covar_min': 0.0002,
                      'covar_max': 0.002,
                      'stationary_prob': 0.4,
                      'random_state': params['random_seed']+entity_idx**2}

        X, y, time_stamps = synthetic_CC_hmm(single_params)
        print(len(time_stamps))

        dist_mat = sdc.compute_dist_mat(np.array(X)) 
        sim_mat = sdc.sim_dist_convert(dist_mat)
        sim_dict = sdc.mat_to_dict(sim_mat, no_zeros=True)

        idx_to_id_dict = dict([[i, i] for i in range(len(y))])
        cluster_id_maps = make_id_cluster_maps(y, idx_to_id_dict)
        id_to_cluster_map, cluster_to_id_map = cluster_id_maps
    
        if time_stamps is None:
            entity_dict = {'cluster_to_id_map': cluster_to_id_map,
                         'id_to_cluster_map': id_to_cluster_map,
                         'index': entity_idx,
                         'similarity_graph': sim_dict}
        else:
            entity_dict = {'cluster_to_id_map': cluster_to_id_map,
                         'id_to_cluster_map': id_to_cluster_map,
                         'index': entity_idx,
                         'similarity_graph': sim_dict,
                         'time_stamps': time_stamps}

        whole_data[entity_idx] = entity_dict

    return whole_data


def make_dataset(hyper_params={}):
    """This function is a dummy function that gets the coordinates and labeles and returns them as data.

    Args:
        hyper_params: A dictionary that has to contain the values for the key 'X' and optionaly for the key 'y'. 
        They function will return the value of 'X' as is. If 'y' is not provided a vector of zeros with the same 
        size as X will be returned. The purpose of this function is to have an identical UI as the other functions
        that generate data.
    """
    params = {'X': np.eye(5)}
    params.update(hyper_params)

    X = params['X']

    if 'y' in params:
        y = params['y']
    else:
        y = np.zeros(len(X), dtype=int)

    return X, y


if __name__ == "__main__":
    import visualize as viz

    # X = generate_data(10, 4711)
    # print("X.shape:", X.shape)
    # plt.scatter(X[:,0], X[:,1])
    # plt.savefig("Gaussian_simple.png", dpi=350)


    # import visualize as viz
    # X, y = synthetic_CC_hmm()
    # viz.viz_2d(X, y, 'synthetic_hmm', do_pca=False)
    # print(y)


    time_stamp_params = {'period':30,
                        'a': 0.4,
                        'max_average':120,
                        'start_date': datetime.datetime(2020,7,18,0,0,0,0)}

    params = {'n_samples': 90,
              'CC_count': 11,
              'period': 30,
              'mean_min': -10.0,
              'mean_max': 10.0,
              'covar_min': 0.05,
              'covar_max': 2.5,
              'stationary_prob': 0.5,
              'random_state': 1234,
              'time_stamp_params':time_stamp_params}

    X, y, ts = synthetic_CC_hmm(params)
    viz.viz_2d(X, y, 'synthetic_hmm_time_stamp', do_pca=False)
    print(y)
    print(ts)
    # print([np.datetime64(elem) for elem in ts])
    # exit(0)    


    from main import dump_dict_to_yaml
    synthetic_data = make_synthetic_data()
    dump_dict_to_yaml(synthetic_data, 'synthetic_hmm_data')

