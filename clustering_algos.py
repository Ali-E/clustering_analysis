"""This module contains various batch clustering function."""
import numpy as np
import pandas as pd
from sklearn import cluster
from sim_dist_coords import compute_dist_mat
from sim_dist_coords import get_coords_equation
from sim_dist_coords import get_coords_mds


np.random.seed(0)


def K_means(coords, hyper_params={}):
    params = {'n_clusters': 2} # default values
    params.update(hyper_params)
    clustering_obj = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    clustering_obj.fit(coords)
    y_pred = clustering_obj.labels_.astype(np.int)
    return y_pred
    

def HAC_single(coords, hyper_params={}): 
    params = {'n_clusters': 2,
            'distance_threshold': None,
            'affinity': 'euclidean'}  # default values
    params.update(hyper_params)
    clustering_obj = cluster.AgglomerativeClustering(linkage='single',
                            n_clusters=params['n_clusters'],
                            affinity=params['affinity'],
                            distance_threshold=params['distance_threshold'])
    clustering_obj.fit(coords)
    y_pred = clustering_obj.labels_.astype(np.int)
    return y_pred


def HAC_complete(coords, hyper_params={}): 
    params = {'n_clusters': 2,
            'distance_threshold': None,
            'affinity': 'euclidean'}  # default values
    params.update(hyper_params)
    clustering_obj = cluster.AgglomerativeClustering(linkage='complete',
                            n_clusters=params['n_clusters'],
                            affinity=params['affinity'],
                            distance_threshold=params['distance_threshold'])
    clustering_obj.fit(coords)
    y_pred = clustering_obj.labels_.astype(np.int)
    return y_pred


def HAC_ward(coords, hyper_params={}): 
    params = {'n_clusters': 2,
            'distance_threshold': None,
            'affinity': 'euclidean'}  # default values
    params.update(hyper_params)
    clustering_obj = cluster.AgglomerativeClustering(linkage='ward',
                            n_clusters=params['n_clusters'],
                            affinity=params['affinity'],
                            distance_threshold=params['distance_threshold'])
    clustering_obj.fit(coords)
    y_pred = clustering_obj.labels_.astype(np.int)
    return y_pred


def HAC_average(coords, hyper_params={}): 
    params = {'n_clusters': 2,
            'distance_threshold': None,
            'affinity': 'euclidean'}  # default values
    params.update(hyper_params)
    clustering_obj = cluster.AgglomerativeClustering(linkage='average',
                            n_clusters=params['n_clusters'],
                            affinity=params['affinity'],
                            distance_threshold=params['distance_threshold'])
    clustering_obj.fit(coords)
    y_pred = clustering_obj.labels_.astype(np.int)
    return y_pred


def DBSCAN(coords, hyper_params={}):
    params = {'eps': 0.1,
            'min_samples': 5}
    params.update(hyper_params)
    clustering_obj = cluster.DBSCAN(eps=params['eps'],
                                    min_samples=params['min_samples'])
    clustering_obj.fit(coords)
    y_pred = clustering_obj.labels_.astype(np.int)
    return y_pred


def identity(coords, hyper_params={}):
    params = {'y': [0 for i in range(len(coords))]}
    params.update(hyper_params)
    return params['y']


def affinity_clustering(dist_mat, hyper_params={}):
    params = {'dist_thresh': 1.5, 
            'p':1.0}
    params.update(hyper_params)
    dist_thresh = params['dist_thresh']
    p = param['p']

    # Make a copy
    dist_matrix_copy = np.array(dist_matrix)
    
    # Fill the diagonal with infinity
    np.fill_diagonal(dist_matrix_copy, np.inf)
    
    # Find the closest neighbor to each node
    closest = np.argmin(dist_matrix_copy, axis=0)
    
    # List of (outgoing) links 
    # (Flip a random coin for creating an outgoing link)
    links = dict([(s, d) if dist_matrix[s, d] < dist_thresh and random.uniform(0, 1) <= p else (s, s)
             for s, d in enumerate(closest)])

    # List of reverse (incoming) links
    reverse_links = collections.defaultdict(set)
    for s, d in links.iteritems():
        reverse_links[d].add(s)
    
    # Do DFS for links to build a graph
    clusters = zip(enumerate(range(dist_matrix.shape[0])))
    while links:
        # Pop a random link
        s, d = links.popitem()
        # Add outgoing link to explore
        to_explore = set([d])
        # Add incoming link(s) to explore
        if s in reverse_links:
            to_explore |= reverse_links[s]
            del reverse_links[s]
        connected_component = set([s]) | to_explore
        while to_explore:
            # Pop a random node to explore
            node = to_explore.pop()
            # Add outgoing links to to_explore 
            if node in links:
                to_explore.add(links[node])
                del links[node]
            # Add incoming links to to_explore 
            if node in reverse_links:
                to_explore |= reverse_links[node]
                del reverse_links[node]
            # Update the cluster
            connected_component |= to_explore
        # Index of connected is set to the node with maximum index
        connected_component_idx = max(connected_component)
        for c in connected_component:
            clusters[c] = connected_component_idx
    return clusters


"""
def iterate_affinity(dist_mat, hyper_params={}):
    params = {'dist_thresh': 1.5, 
             'current_clusters': None,
             'p'=1.0}
    params.update(hyper_params)
    dist_thresh = params['dist_thresh']
    current_clusters = params['current_clusters']
    p = param['p']

    def ave_link_func(nodes1, nodes2):
        result = sum(orig_dist_matrix[n1, n2] for n1 in nodes1 for n2 in nodes2)
        return result / (len(nodes1) * len(nodes2))

    # Mapping from [0, ..., current_size - 1] to indexes of original clusters
    idx_to_cluster_id = dict(enumerate(sorted(list(set(current_clusters)))))
    cluster_id_to_idx = {v: k for k, v in idx_to_cluster_id.iteritems()}
    current_size = len(set(current_clusters))
    next_dist_matrix = np.zeros((current_size, current_size))
    
    cluster_id_to_nodes = collections.defaultdict(set)
    for node, cluster in enumerate(current_clusters):
        cluster_id_to_nodes[cluster].add(node)
    
    # Compute the next distance matrix
    for i in xrange(current_size):
        for j in xrange(current_size):
            next_dist_matrix[i, j] = ave_link_func(cluster_id_to_nodes[idx_to_cluster_id[i]], 
                                                   cluster_id_to_nodes[idx_to_cluster_id[j]])
            
    # Compute the next set of clusters
    next_clusters = affinity_clustering(next_dist_matrix, dist_thresh, p)
    
    # Flatten the cluster list
    next_clusters_flattened = [next_clusters[cluster_id_to_idx[current_cluster_idx]]
                               for current_cluster_idx in current_clusters]
    
    return next_clusters_flattened
"""
