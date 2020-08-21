"""The main module continas functions for reading the data,
running the clustering methods using the unified API
on the data, and dumping the results in a specific format."""
import numpy as np
import sim_dist_coords as sdc
import yaml


def parse_input_file(yaml_filename, multi_entry=False):
    """Reads a yaml file.
    
    Returns:
        A dictionary object that contains the mapping
        in the yaml file. 
    """
    with open(yaml_filename) as file:
        mapping = yaml.load(file, Loader=yaml.FullLoader)
    if multi_entry:
        return list(mapping)
    return mapping


def dump_dict_to_yaml(mapping, filename):
    """Writes the mapping to a yaml file.
    
    Args:
        mapping: A dictionary object
        filename: The name of the yaml file that the mapping
            will be written to. The name should not contain 
            '.yaml' postfix; it will be added in the function.
    """
    with open(filename + '.yaml', 'w') as file:
        yaml.dump(mapping, file)


def make_id_cluster_maps(y_pred, idx_to_id_dict):
    """Given the assignments for a set of data-points returns the mappings between entity ids and clusters.

    Args:
        y_pred: A list of numbers that contains the predictions of a clustering algorithm.
            The i-th elements value in this list is the cluster_id that was assigned to
            the i-th element (idx=i-1).
        idx_to_id_dict: It is a mapping for which the values are the original ids and keys are
            the row numbers in the corresponding coordinate matrix. It will be an identity 
            mapping if the original input is already a coordinate matrix.

    Returns:
        id_to_cluster_dict: This is a mapping from the original ids to the cluster that
            it was assigned to accoriding to y_pred.
        cluster_to_id_dict: This id a ampping in which the cluster ids are the keys and
            the values are the list of the original ids of the members of the clusters.
    """
    id_to_cluster_dict = {}
    cluster_to_id_dict = {}

    for idx, cluster_id in enumerate(y_pred):
        curr_id = idx_to_id_dict[idx]
        id_to_cluster_dict[curr_id] = cluster_id

        if cluster_id in cluster_to_id_dict:
            cluster_to_id_dict[cluster_id].append(curr_id)
        else:
            cluster_to_id_dict[cluster_id] = [curr_id]

    return id_to_cluster_dict, cluster_to_id_dict


def fix_read_labeles_order(y, id_to_idx_dict, idx_sorted_pos_dict=None):
    """Given a list of labels ordered according to sorted ids, rearranges the list based on sorted idx.

    Args:
        y: The label list generated by a clustering algorithm. The i-th value corresponds to the i-th
            id when they are put in a sorted order.
        id_to_idx_dict: It is a mapping for which the keys are the original ids that the labels are 
            sorted based on. Each of them maps to another set of idx values that we want to rearrange
            the labels based on their sorted order.
        idx_sorted_pos_dict: If the idx values are not integers (e.g. string or any other object), 
            this dictionary should be given as input to show the position of each idx in their 
            sorted order. 

    Returns:
        Returns the rearranged labels in which they value at i-th position corresponds to the i-th
        idx when they are put in a sorted order.
    """
    rearranged_y = np.zeros(len(y))
    sorted_ids = sorted(id_to_idx_dict.keys()) 

    for id_index, id in enumerate(sorted_ids):
        corres_idx = id_to_idx_dict[id]
        
        if idx_sorted_pos_dict is not None:
            corres_idx = idx_sorted_pos_dict[corres_idx]

        rearranged_y[corres_idx] = y[id_index]

    return rearranged_y


def ext_labels_list_from_dict(idx_to_cluster_dict):
    """Given a map of idx to cluster ids, generates labels list.
    
    Args:
        idx_to_cluster_dict: A dictionary in which the keys are the indices 
            and the values are cluster ids.

    Returns:
        labels: A list in which the value at index i, is the cluster id that
            the i-th element belongs to.
    """
    labels = []
    for idx in idx_to_cluster_dict:
        labels[idx] = idx_to_cluster_dict[idx]
    return labels


def create_sample_files():
    """This function creates sample files for verification of the pipeline. 
    
    It uses synthetis_CC function to create sample data points and labels.
    Then is computes the distances and makes the distance matrix and converts
    that to similarity matrix. The keys of the similarity matrix are tweaked 
    to try a more realistic pipeline. The new keys are strings and their sorted
    order represents the order of arrival of the datapoints. That is why the 
    previously computed labeles have to be rearranged before saving them into 
    a file. Other generated files contain original coordinates and a 2d plot 
    of them in a pca-transformed 2d space. 
    """
    X, y = synthetic_CC({'n_samples': 105, 'centers': 11})
    dist_mat = sdc.compute_dist_mat(X)

    sim_mat = sdc.sim_dist_convert(dist_mat, normalize_flag=True)
    sim_dict = sdc.mat_to_dict(sim_mat)
    sim_dict_with_ids = {}

    for key in sim_dict:
        new_key = ('id' + str(key[0]), 'id' + str(key[1]))
        sim_dict_with_ids[new_key] = sim_dict[key]

    all_new_ids = sdc.get_all_ids(sim_dict_with_ids)
    all_new_ids_pos_dict = dict(list(zip(all_new_ids, range(len(all_new_ids)))))

    reg_ids = sdc.get_all_ids(sim_dict)
    changed_ids = ['id' + str(id) for id in reg_ids]
    key_map_tups = zip(reg_ids, changed_ids)
    key_map = dict(list(key_map_tups))

    y_reordered = fix_read_labeles_order(y, key_map, idx_sorted_pos_dict=all_new_ids_pos_dict)

    labels_dict = {}
    for i in range(len(all_new_ids)):
        labels_dict['id'+str(i)] = int(y[i])

    # viz.viz_2d(X, y_reordered, 'synthetic_CC')
    dump_dict_to_yaml(sim_dict_with_ids, 'synthetic_CC')
    dump_dict_to_yaml(labels_dict, 'synthetic_CC_labels')
    np.savetxt('synthetic_CC_labels.csv', y_reordered)
    np.savetxt('synthetic_CC_coords.csv', X)


def clustering_method_call(input_data, clustering_method, hyper_params={}):
    """Runs a clustering method on a similarity dictionary.

    Args:
        input_data: this is the input that will be passed to the method
        function. It is in either of the formats below:
            similarity_dict: A dictionary object in which the keys
                are tuple of a pair of ids of the objects that are
                being clustered. The values in this dictionary are
                the similarity values.
            data_vectors: In this format, the data is a 2d-array 
                in which each row contains the vector representation
                of one of the samples.
        method: The name of a function that contains the implemation
            of a clustering algorithm.
        hyper_parameters: An optional dictionary object that contains two
            values that is used in this function to prepare the input_data
            to be passed to the method. It also contains
            the valuefor the hyper-parameters that are being used
            by the clustering method. (default: an embety dictionary).
            The two hyperparameters that are used in this function are:
                input_format: This variable defines the format of the input_data.
                    At this point it can be set to 'similarity_dict' or 'data_vectors'.
                    (default: 'similarity_dict')
                required_format: This variable defines the format of data required by
                    the method. At this point it can be set to 'similarity_mat' or 
                    'data_vectors' (default: 'similarity_mat').
        
    Returns:
        The function returns a dictionary that contains 4 keys and corresponding object:
        coords: coordinate of the data points in the Euclidean space.
        y_pred: labels assigned to each data point by the clustering method.
        id_to_idx_dict: mapping of original ids of the point to their index in the
            coords matrix or y_pred array.
        idx_to_id_dict: reverse mapping of the one stored in id_to_idx_dict
        id_to_cluster_dict: mapping of original ids of the objects to the id of their
            assigned cluster
        cluster_to_id_dict: mapping of cluster ids to the id of their constituting points.
    """
    params = {'input_format': 'similarity_dict',
            'required_format': 'data_vectors',
            'precision_boost': 'AUTO',
            'top_eigenvals': None,
            'data_size': None,
            'embedding_method': sdc.get_coords_mds_stress,
            'coords_given': None,
            'compute_coords': False,
            'max_sim_value': 1.0,
            'embedding_hyper_params': {}}
    params.update(hyper_params)

    input_format = params['input_format']
    required_format = params['required_format']
    compute_coords = params['compute_coords']
    coords_given = params['coords_given']
    max_sim_value = params['max_sim_value']
    data_size = params['data_size']

    if coords_given is not None:
        compute_coords = False

    if input_format == 'similarity_dict' and required_format == 'data_vectors':
        if coords_given is None:
            coords, id_to_idx_dict, idx_to_id_dict = sdc.sim_dict_to_coords(input_data,
                                                    data_size=params['data_size'],
                                                    top_eigenvals=params['top_eigenvals'],
                                                    embedding_method=params['embedding_method'],
                                                    hyper_params=params['embedding_hyper_params'])
        else:
            coords = coords_given
            id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
            idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))

        y_pred = clustering_method(coords, hyper_params)

    elif input_format == 'similarity_dict' and required_format == 'similarity_mat': 
        if coords_given is None:
            coords, sim_mat, id_to_idx_dict, idx_to_id_dict = sdc.sim_dict_to_coords(input_data,
                                                            data_size=params['data_size'],
                                                            return_sim_mat=True,
                                                            top_eigenvals=params['top_eigenvals'],
                                                            embedding_method=params['embedding_method'],
                                                            hyper_params=params['embedding_hyper_params'])
        else:
            coords = coords_given
            id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
            idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))

        y_pred = clustering_method(sim_mat, hyper_params)

    elif input_format == 'data_vectors' and required_format == 'similarity_mat': 
        dist_mat = sdc.compute_dist_mat(input_data)
        sim_mat = sdc.sim_dist_convert(dist_mat)
        y_pred = clustering_method(sim_mat, hyper_params)
        id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
        idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))
        coords = input_data
        
    elif input_format == 'data_vectors' and required_format == 'data_vectors': 
        y_pred = clustering_method(input_data, hyper_params)
        id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
        idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))
        coords = input_data

    elif input_format == 'similarity_dict' and required_format == 'dist_mat':
        # print(max_sim_value)
        sim_mat = sdc.dict_to_mat(input_data, data_size=params['data_size'], max_val=max_sim_value)[0]
        dist_mat = sdc.sim_dist_convert(sim_mat, normalize_flag=False)
        hyper_params['affinity'] = 'precomputed'
        hyper_params['n_clusters'] = None
        id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
        idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))
        y_pred = clustering_method(dist_mat, hyper_params)
        coords = coords_given
        if compute_coords:
            coords, sim_mat, id_to_idx_dict, idx_to_id_dict = sdc.sim_dict_to_coords(input_data,
                                                            data_size=params['data_size'],
                                                            return_sim_mat=True,
                                                            top_eigenvals=params['top_eigenvals'],
                                                            embedding_method=params['embedding_method'],
                                                            hyper_params=params['embedding_hyper_params'])

    elif input_format == 'dist_mat' and required_format == 'dist_mat':
        hyper_params['affinity'] = 'precomputed'
        hyper_params['n_clusters'] = None
        id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
        idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))
        y_pred = clustering_method(input_data, hyper_params)
        coords = coords_given
        if compute_coords:
            coords, sim_mat, id_to_idx_dict, idx_to_id_dict = sdc.sim_dict_to_coords(input_data,
                                                            data_size=params['data_size'],
                                                            return_sim_mat=True,
                                                            top_eigenvals=params['top_eigenvals'],
                                                            embedding_method=params['embedding_method'],
                                                            hyper_params=params['embedding_hyper_params'])

    elif input_format == 'data_vectors' and required_format == 'similarity_dict': 
        dist_mat = sdc.compute_dist_mat(input_data)
        sim_mat = sdc.sim_dist_convert(dist_mat)
        sim_dict = sdc.mat_to_dict(sim_mat, no_zeros=True)
        y_pred = clustering_method(sim_dict, hyper_params)
        id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
        idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))
        coords = input_data


    elif input_format == 'similarity_dict' and required_format == 'similarity_dict':
        y_pred = clustering_method(input_data, hyper_params)
        id_to_idx_dict = dict(list(zip(range(data_size), range(data_size))))
        idx_to_id_dict = dict(list(zip(range(data_size), range(data_size))))
        coords = coords_given
        if compute_coords:
            coords, sim_mat, id_to_idx_dict, idx_to_id_dict = sdc.sim_dict_to_coords(input_data,
                                                            data_size=params['data_size'],
                                                            return_sim_mat=True,
                                                            top_eigenvals=params['top_eigenvals'],
                                                            embedding_method=params['embedding_method'],
                                                            hyper_params=params['embedding_hyper_params'])
    
    else:
        print('input_format or required_format not set correctly!')

    id_cluster_maps = make_id_cluster_maps(y_pred, idx_to_id_dict)

    output_dict = {'coords': coords,
            'y_pred': y_pred,
            'id_to_idx_map': id_to_idx_dict,
            'idx_to_id_map': idx_to_id_dict,
            'id_to_cluster_map': id_cluster_maps[0],
            'cluster_to_id_map': id_cluster_maps[1]}

    return output_dict


def grid_search(input_data, target, method, hyper_params):
    params = {'input_format': 'similarity_dict',
            'required_format': 'data_vectors',
            'precision_boost': 'AUTO',
            'data_size': None}
    params.update(hyper_params)

    gs = GridSearchCV(method, params)
    gs.fit(input_data, target)

    return gs.cv_results_


if __name__ == "__main__":
    # import sys
    # from clustering_algos import *
    # from dataset import *
    # import visualize as viz


    """
    input_file = sys.argv[1]  #contains the similarity mapping
    similarity_map = parse_input_file(input_file)
    hyper_params = {'linkage': 'single'}

    output_mappings = clustering_method_call(similarity_map, 
                                             calg.agglomerative_algos, 
                                             hyper_params)
    cluster_to_entity_map, entity_to_cluster_map = output_mappings

    dump_dict_to_yaml(cluster_to_entity_map, 'cluster_to_entity_map')
    dump_dict_to_yaml(entity_to_cluster_map, 'entity_to_cluster_map')
    """


    """
    all_methods = [('K_means', K_means, {'required_format': 'data_vectors'}),
            ('single', HAC_single, {'required_format': 'data_vectors'}),
            ('complete', HAC_complete, {'required_format': 'data_vectors'}),
            ('ward', HAC_ward, {'required_format': 'data_vectors'}),
            ('average', HAC_average, {'required_format': 'data_vectors'}),
            ('identity', identity, {'required_format': 'data_vectors'})]

    all_datasets = [(noisy_circle, {'factor': 0.5, 'noise': 0.05}, {'input_format': 'data_vectors'}),
            (noisy_circle, {}, {'input_format': 'data_vectors'}),
            (noisy_moons, {}, {'input_format': 'data_vectors'}),
            (blobs, {'n_samples': 32, 'centers': 5, 'random_state': 142}, {'input_format': 'data_vectors', 'n_clusters': 5}),
            (synthetic_CC, {'n_samples': 103, 'centers': 9}, {'input_format': 'data_vectors', 'n_clusters': 9}),
            (synthetic_CC, {'n_samples': 154, 'centers': 13, 'transformation': [[0.8, -0.5], [-0.4, 0.6]]},
                    {'input_format': 'data_vectors', 'n_clusters': 13})]

    viz.compare_multi(all_methods, all_datasets[3:], 'compare_all_methods.png')
    """

