"""Contains different functions and modules 
for incremental clustering methods."""
from copy import deepcopy
import datetime
import heapq
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import time

from clustering_algos import *
from dataset import *
from main import clustering_method_call
from sim_dist_coords import *
import visualize as viz


class sessioonDist:
    def __init__(self, idx, dist):
        self.idx = idx
        self.dist = dist

    def __lt__(self, other):
        return self.dist > other.dist

    def __str__(self):
        return '(' + str(self.idx) + ', ' + str(self.dist) + ')'

    def __repr__(self):
        return self.__str__()


def get_first(heap, removed_set, last_acceptable_dist):
    if heap == None or len(heap) < 1:
        return None
    while heap[0].idx in removed_set and (heap[0].idx not in last_acceptable_dist or (last_acceptable_dist[heap[0].idx] != heap[0].dist and last_acceptable_dist[heap[0].idx] == -1)):
        print('found!')
        heapq.heappop(heap)
    return heap[0]


def avg_dist(i, k, count_list, vsum_list):
    dot_product_of_sum = np.dot((vsum_list[i] + vsum_list[k]), (vsum_list[i] + vsum_list[k]))
    return (1.0/((count_list[i] + count_list[k])*(count_list[i] + count_list[k] - 1))) * (dot_product_of_sum - (count_list[i] + count_list[k]))
    

def average_HAC(coords, hyper_params={}):
    sim_mat = sdc.compute_dot_mat(coords)
    print('sim_mat\n', sim_mat)
    heap_list = []
    # not_merged = np.ones(len(sim_mat), dtype=int)
    not_merged = [True for i in range(len(sim_mat))]
    removed_set = [set([i]) for i in range(len(sim_mat))]
    last_acceptable_dist = [{} for i in range(len(sim_mat))]
    # removed_set[0].add(1)
    count_list = np.ones(len(sim_mat), dtype=int)
    vsum_list = [np.array(deepcopy(vect)) for vect in coords]
    
    for i in range(len(sim_mat)):
        dist_list = [sessioonDist(j, float(sim_mat[i][j])) for j in range(len(sim_mat))]
        heapq.heapify(dist_list)
        heap_list.append(dist_list)

    clusters = ''
    for iteration in range(len(sim_mat)-1):
        print('-----------------------')
        print('not merged\n', not_merged)
        remaining_indices = np.arange(len(sim_mat))
        remaining_indices = remaining_indices[not_merged]
        print('remaining_indices\n', remaining_indices)
        
        print('heap_list\n', heap_list)
        print('last_acceptable_dist\n', last_acceptable_dist)
        remaining_min_dists = np.array([get_first(curr_heap, removed_set[idx], last_acceptable_dist[idx]) for idx, curr_heap in enumerate(heap_list)])[not_merged]
        print('remaining_min_dists\n', remaining_min_dists)
        
        min_tups = zip(remaining_indices, remaining_min_dists)
        min_min = min(min_tups, key=lambda x: x[1])
        k1 = min_min[0]
        k2 = min_min[1].idx

        print(k1, k2)

        count_list[k1] = count_list[k1] + count_list[k2]
        # vsum_list[k1] = preprocessing.normalize(np.array([vsum_list[k1] + vsum_list[k2]]), norm='l2')[0]
        vsum_list[k1] = np.array([vsum_list[k1] + vsum_list[k2]])[0]
        print('count_list: ', count_list)
        print('vsum_list: ', vsum_list)
        not_merged[k2] = False
        new_heap = []
        
        for i in range(len(sim_mat)):
            if not_merged[i] == False or i == k1 or i == k2:
                continue
            new_dist = avg_dist(i, k1, count_list, vsum_list)
            removed_set[i].add(k1)
            last_acceptable_dist[i][k1] = new_dist
            removed_set[i].add(k2)
            last_acceptable_dist[i][k2] = -1

            heapq.heappush(heap_list[i], sessioonDist(k1, new_dist))
            heapq.heappush(new_heap, sessioonDist(i, new_dist))
            heap_list[k1] = new_heap


def average_dist(coords, cluster, point):
    sum_dist = 0.0
    count = 0
    for idx in list(cluster):
        new_point = coords[idx]
        new_dist = np.linalg.norm(np.array(point)-np.array(new_point))
        # print('dist: ', idx, point, ' =', new_dist)
        sum_dist += new_dist
        count += 1
    cluster_list = list(cluster)
    for i in range(len(cluster_list)):
        idx = cluster_list[i]
        for j in range(i):
            idx_2 = cluster_list[j]
            point = coords[idx_2]
            new_point = coords[idx]
            new_dist = np.linalg.norm(np.array(point)-np.array(new_point))
            sum_dist += new_dist
            count += 1
    return sum_dist/count


def compute_new_dist(coords, cluster, cluster_2):
    sum_dist = 0.0
    count = 0
    for idx in list(cluster):
        for idx_2 in list(cluster_2):
            point = coords[idx_2]
            new_point = coords[idx]
            new_dist = np.linalg.norm(np.array(point)-np.array(new_point))
            sum_dist += new_dist
            count += 1
    cluster_list = list(cluster)
    for i in range(len(cluster_list)):
        idx = cluster_list[i]
        for j in range(i):
            idx_2 = cluster_list[j]
            point = coords[idx_2]
            new_point = coords[idx]
            new_dist = np.linalg.norm(np.array(point)-np.array(new_point))
            sum_dist += new_dist
            count += 1
    cluster_2_list = list(cluster_2)
    for i in range(len(cluster_2_list)):
        idx = cluster_2_list[i]
        for j in range(i):
            idx_2 = cluster_2_list[j]
            point = coords[idx_2]
            new_point = coords[idx]
            new_dist = np.linalg.norm(np.array(point)-np.array(new_point))
            sum_dist += new_dist
            count += 1
    return sum_dist/count


def incremental_average(coords, hyper_params={}):
    params = {'dist_threshold': 1.0}
    params.update(hyper_params)
    dist_threshold = params['dist_threshold']

    clusters = [set([0])]
    cluster_dist = {}

    for i in range(1, len(coords)):
        best_cluster = -1
        min_dist = float('inf')
        for idx, cluster in enumerate(clusters):
            new_dist = average_dist(coords, cluster, coords[i])
            if new_dist < min_dist:
                min_dist = new_dist
                best_cluster = idx 
        # print('min dist: ', min_dist)
        # print('best cluster: ', best_cluster)
        if min_dist < dist_threshold:
            changed_cluster = idx
            clusters[best_cluster].add(i)
        else:
            changed_cluster = len(clusters)
            clusters.append(set([i]))

        # print(clusters)

        remaining_flag = True
        while remaining_flag:
            remaining_flag = False
            for c in range(len(clusters)):
                if c == changed_cluster:
                    continue
                cc_dist = compute_new_dist(coords, clusters[changed_cluster], clusters[c])
                if cc_dist < dist_threshold:
                    remaining_flag = True
                    clusters[changed_cluster] = clusters[changed_cluster] | clusters[c]
                    # print(clusters)
                    # print(c)
                    # clusters.pop[c]
                    if c+2 > len(clusters):
                        clusters = clusters[:c]
                    else:
                        clusters = clusters[:c] + clusters[c+1:]
                    if c < changed_cluster:
                        changed_cluster -= 1
                    break
        # print('clusters after:\n', clusters)

    # print(clusters)
    y = [-1 for i in range(len(coords))]
    for idx, cluster in enumerate(clusters):
        for elem in list(cluster):
            y[elem] = idx
    return y

                
def hierarchical_average(coords, hyper_params={}):
    params = {'dist_threshold': 1.0}
    params.update(hyper_params)
    dist_threshold = params['dist_threshold']

    clusters = [set([0])]
    cluster_dist = {}

    for i in range(1, len(coords)):
        best_cluster = -1
        min_dist = 999999999999
        for idx, cluster in enumerate(clusters):
            new_dist = average_dist(coords, cluster, coords[i])
            if new_dist < min_dist:
                min_dist = new_dist
                best_cluster = idx 
        if min_dist < dist_threshold:
            changed_cluster = idx
            clusters[best_cluster].add(i)
        else:
            changed_cluster = len(clusters)
            clusters.append(set([i]))

        remaining_flag = True
        while remaining_flag:
            remaining_flag = False
            for c in range(len(clusters)):
                if c == changed_cluster:
                    continue
                cc_dist = compute_new_dist(coords, clusters[changed_cluster], clusters[c])
                if cc_dist < dist_threshold:
                    remaining_flag = True
                    clusters[changed_cluster] = clusters[changed_cluster] | clusters[c]
                    # print(clusters)
                    # print(c)
                    # clusters.pop[c]
                    if c+2 > len(clusters):
                        clusters = clusters[:c]
                    else:
                        clusters = clusters[:c] + clusters[c+1:]
                    if c < changed_cluster:
                        changed_cluster -= 1
                    break

    # print(clusters)
    y = [-1 for i in range(len(coords))]
    for idx, cluster in enumerate(clusters):
        for elem in list(cluster):
            y[elem] = idx
    return y


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def average_dist_SI(dist_mat, cluster, point, outdated_set={}, linkage='centroid'):
    sum_dist = 0.0
    count = 0
    cluster_list = list(cluster)
    for idx in cluster_list:
        if idx in outdated_set:
            continue

        new_dist = dist_mat[point, idx]
        sum_dist += new_dist
        count += 1

    if linkage == 'average':
        for i in range(len(cluster_list)):
            idx = cluster_list[i]
            if idx in outdated_set:
                continue

            for j in range(i):
                idx_2 = cluster_list[j]
                if idx_2 in outdated_set:
                    continue

                new_dist = dist_mat[idx, idx_2]
                sum_dist += new_dist
                count += 1
    else:
        if linkage != 'centroid':
            print('not a valid linkage function!')
            exit(0)

    if count == 0:
        return float('inf')
    return sum_dist/count


def compute_new_dist_SI(dist_mat, cluster, cluster_2, outdated_set={}, linkage='centroid'):
    sum_dist = 0.0
    count_intra = 0
    count_inter = 0
    for idx in list(cluster):
        if idx in outdated_set:
            continue

        for idx_2 in list(cluster_2):
            if idx_2 in outdated_set:
                continue

            new_dist = dist_mat[idx, idx_2]
            sum_dist += new_dist
            count_inter += 1
    
    if count_inter == 0:
        return float('inf')

    if linkage == 'average':
        cluster_list = list(cluster)
        for i in range(len(cluster_list)):
            idx = cluster_list[i]
            if idx in outdated_set:
                continue

            for j in range(i):
                idx_2 = cluster_list[j]
                if idx_2 in outdated_set:
                    continue

                new_dist = dist_mat[idx, idx_2]
                sum_dist += new_dist
                count_intra += 1

        cluster_2_list = list(cluster_2)
        for i in range(len(cluster_2_list)):
            idx = cluster_2_list[i]
            if idx in outdated_set:
                continue

            for j in range(i):
                idx_2 = cluster_2_list[j]
                if idx_2 in outdated_set:
                    continue

                new_dist = dist_mat[idx, idx_2]
                sum_dist += new_dist
                count_intra += 1

    else:
        if linkage != 'centroid':
            print('not a valid linkage function!')
            exit(0)

    return sum_dist/(count_intra + count_inter)


def incremental_average_SI(sim_map, hyper_params={}):
    params = {'dist_threshold': 0.5,
              'update_clusters': False,
              'clusters': [set([0])],
              'time_stamps': None,
              'window_size': 15,
              'data_size': None,
              'linkage': 'centroid',
              'max_sim_value': 1.0,
              'only_final_clusters': False,
              'normalize_flag': False,
              'keep_timestamp_results': {'active': False, 'v_scores': [], 'batch_params': {}, 'complete_batch': [], 'v_scores_complete': [],
                                          'size_list': [], 'h_scores': [], 'h_scores_complete': [], 'c_scores': [], 'c_scores_complete': [],
                                           'v_scores_limited': []},
              'dist_mat': False}
    params.update(hyper_params)

    data_size = params['data_size']
    max_sim_value = params['max_sim_value']
    normalize_flag = params['normalize_flag']
    linkage = params['linkage']
    keep_timestamp_results = params['keep_timestamp_results']

    if params['dist_mat']:
        dist_mat = sim_map
    else:
        sim_mat = dict_to_mat(sim_map, data_size=data_size, max_val=max_sim_value)[0]
        dist_mat = sim_dist_convert(sim_mat, normalize_flag=normalize_flag)

    dist_threshold = params['dist_threshold']
    update_clusters = params['update_clusters']
    clusters = deepcopy(params['clusters'])

    time_stamps_orig = params['time_stamps']
    window_size = params['window_size']
    windowed_flag = False

    if time_stamps_orig is not None and window_size is not None:
        windowed_flag = True
        try:
            time_stamps = [np.datetime64(datetime.datetime.utcfromtimestamp(elem)) for elem in time_stamps_orig]
        except:
            time_stamps = [np.datetime64(elem) for elem in time_stamps_orig]

    outdated_set = set()
    already_clustered = set()
    for cluster in clusters:
        already_clustered |= cluster

    first_window_move_flag = True
    first_window_move_idx = 0
    outdate_index = 0  # points to position after the last outdated element.
    for i in range(len(dist_mat)):
        if i in already_clustered:
            continue

        if windowed_flag:
            while (time_stamps[i] - time_stamps[outdate_index]) / np.timedelta64(86400000000, 'us') >= window_size:
                print((time_stamps[i] - time_stamps[outdate_index]) / np.timedelta64(86400000000, 'us'))
                outdated_set.add(outdate_index)
                outdate_index += 1
                if first_window_move_flag:
                    first_window_move_flag = False
                    first_window_move_idx = i

        best_cluster = -1
        min_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            new_dist = average_dist_SI(dist_mat, cluster, i, outdated_set=outdated_set, linkage=linkage)
            if new_dist < min_dist:
                min_dist = new_dist
                best_cluster = idx 

        update_necessary = False
        if min_dist < dist_threshold:
            changed_cluster = idx
            clusters[best_cluster].add(i)
            update_necessary = True
        else:
            changed_cluster = len(clusters)
            clusters.append(set([i]))

        if update_clusters and update_necessary:
            remaining_flag = True
            while remaining_flag:
                remaining_flag = False
                for c in range(len(clusters)):
                    if c == changed_cluster:
                        continue
                    cc_dist = compute_new_dist_SI(dist_mat, clusters[changed_cluster], clusters[c], outdated_set=outdated_set, linkage=linkage)
                    if cc_dist < dist_threshold:
                        remaining_flag = True
                        clusters[changed_cluster] = clusters[changed_cluster] | clusters[c]
                        if c >= len(clusters)-1:
                            clusters = clusters[:c]
                        else:
                            clusters = clusters[:c] + clusters[c+1:]
                        if c < changed_cluster:
                            changed_cluster -= 1
                        break

        if keep_timestamp_results['active']:
            # print('-------------')
            keep_timestamp_results['first_window_move_idx'] = first_window_move_idx
            tmp_y_inc = [-1 for _ in range(i+1)]
            for idx, cluster in enumerate(clusters):
                for elem in list(cluster):
                    tmp_y_inc[elem] = idx
            tmp_y_inc = tmp_y_inc[outdate_index:]
            # print('inc: ', tmp_y_inc)

            tmp_sim_map = filter_sim_map(sim_map, i)
            
            ### if we do not want to outdate:
            keep_timestamp_results['batch_params']['data_size'] = i+1
            keep_timestamp_results['size_list'].append(len(tmp_y_inc))
            tmp_batch_output = clustering_method_call(tmp_sim_map, 
                                                    HAC_average, 
                                                    hyper_params=keep_timestamp_results['batch_params'])
            tmp_y_batch = tmp_batch_output['y_pred']
            tmp_y_batch = tmp_y_batch[-len(tmp_y_inc):]

            keep_timestamp_results['v_scores'].append(v_measure_score(tmp_y_batch, tmp_y_inc)) 
            keep_timestamp_results['h_scores'].append(homogeneity_score(tmp_y_batch, tmp_y_inc)) 
            keep_timestamp_results['c_scores'].append(completeness_score(tmp_y_batch, tmp_y_inc)) 
 
            ### if we want to outdate:
            tmp_sim_map = filter_sim_map(tmp_sim_map, outdate_index, min_case=True)
            keep_timestamp_results['batch_params']['data_size'] = len(tmp_y_inc)
            tmp_batch_output = clustering_method_call(tmp_sim_map, 
                                                    HAC_average, 
                                                    hyper_params=keep_timestamp_results['batch_params'])
            tmp_y_batch = tmp_batch_output['y_pred']

            keep_timestamp_results['v_scores_limited'].append(v_measure_score(tmp_y_batch, tmp_y_inc)) 


            complete_batch = keep_timestamp_results['complete_batch']
            if len(complete_batch) > 0:
                keep_timestamp_results['v_scores_complete'].append(v_measure_score(complete_batch[outdate_index: i+1], tmp_y_inc))
                keep_timestamp_results['h_scores_complete'].append(homogeneity_score(complete_batch[outdate_index: i+1], tmp_y_inc))
                keep_timestamp_results['c_scores_complete'].append(completeness_score(complete_batch[outdate_index: i+1], tmp_y_inc))

    first_index = 0
    if params['only_final_clusters']:
        first_index = outdate_index

    y = [-1 for i in range(len(dist_mat))]
    for idx, cluster in enumerate(clusters):
        for elem in list(cluster):
            if elem < first_index:
                continue
            y[elem] = idx
    return y[first_index:]


def filter_sim_map(sim_map, max_id, min_case=False):
    new_sim_map = {}
    for tup in sim_map:
        if not min_case:
            if tup[0] <= max_id and tup[1] <= max_id:
                new_sim_map[tup] = sim_map[tup]
        else:
            if tup[0] >= max_id and tup[1] >= max_id:
                new_sim_map[(tup[0]-max_id, tup[1]-max_id)] = sim_map[tup]
    return new_sim_map


def warm_start_inc_avg_SI(sim_map, hyper_params={}):
    params = {'batch_portion': 0.5,
              'data_size': None,
              'dist_threshold': 0.1,
              'batch_threshold': 0.9,
              'time_stamps': None,
              'window_size': 15,
              'batch_params': {},
              'inc_params': {}}
    params.update(hyper_params)

    dist_threshold = params['dist_threshold']
    batch_threshold = params['batch_threshold']
    all_data_size = params['data_size']
    window_size = params['window_size']
    time_stamps_orig = params['time_stamps']
    batch_portion = params['batch_portion']
    if time_stamps_orig is not None:
        try:
            time_stamps = [np.datetime64(datetime.datetime.utcfromtimestamp(elem)) for elem in time_stamps_orig]
        except:
            time_stamps = [np.datetime64(elem) for elem in time_stamps_orig]

        period = 1 + int((time_stamps[-1] - time_stamps[0]) / np.timedelta64(86400000000, 'us'))
        batch_period = np.timedelta64(int(np.ceil(period*batch_portion))* 86400000000, 'us')
        batch_size = 0
        for i in range(len(time_stamps)):
            if time_stamps[i] < time_stamps[0] + batch_period:
                batch_size += 1
            else:
                break
    else:
        time_stamps = time_stamps_orig
        batch_size = int(np.ceil(all_data_size * batch_portion))

    batch_sim_map = filter_sim_map(sim_map, batch_size-1)

    batch_params = {'input_format': 'similarity_dict',
                    'required_format': 'dist_mat',
                    'distance_threshold': batch_threshold,
                    'n_clusters': None,
                    'data_size': batch_size,
                    # 'top_eigenvals': 2,
                    # 'embedding_method': get_coords_mds_stress,
                    # 'embedding_hyper_params': {'eps': 0.0000001,
                    #                             'max_iter': 1000}
                    }
    batch_params.update(params['batch_params'])

    output_dict = clustering_method_call(batch_sim_map, 
                                        HAC_average, 
                                        batch_params)
    
    y_pred_batch = output_dict['y_pred']
    clusters_initial_tmp = output_dict['cluster_to_id_map'].values()
    clusters_initial = [set(cluster) for cluster in clusters_initial_tmp]

    inc_params = {'dist_threshold': dist_threshold,
                  'update_clusters': False,
                  'clusters': clusters_initial,
                  'data_size': all_data_size,
                  'time_stamps': time_stamps,
                  'window_size': window_size}
    inc_params.update(params['inc_params'])

    y_label_inc = incremental_average_SI(sim_map, hyper_params=inc_params)
    return y_label_inc, y_pred_batch



if __name__ == "__main__":

    # first = get_first([sessioonDist(2, 49.166666666666664), sessioonDist(1, 34.33333333333333), sessioonDist(2, 0.81373), sessioonDist(3, 0.77396), sessioonDist(1, 0.89443)], {0, 1, 2, 3}, {2: -1, 3: -1, 1: 34.33333333333333})
    # print(first)
    # exit(0)

    a = sessioonDist(2, 5)
    print(a)
    print(a.idx)
    print(a.dist)

    b = sessioonDist(3, 5)
    print(a < b)

    coords = [[1, 1],
              [3, 1],
              [6, 1],
              [10, 1]]

    print('------------')

    """
    coords = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    coords = preprocessing.normalize(coords, norm='l2')

    y = incremental_HAC(coords)
    print(y)

    from sklearn.cluster import AgglomerativeClustering


    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average').fit(coords)
    print(clustering.labels_)
    plot_dendrogram(clustering, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig('new.png')
    """

    coords = [[1,0],
              [2.8,0],
              [1,0.9],
              [2.8,0.9],
              [1.9,0.9],
              [1.9,0],
              [1.0,0.4],
              [1.9,0.4]]

    coords = [[1,0.9],
              [2.8,0.5],
              [1.9,0],
              [1.9,0.4],
              [2.8,0],
              [1.9,0.9],
              [1,0],
              [1.0,0.4]]


    import datetime
    time_stamps = [datetime.datetime(2020,7,18,5,1,3,23),
                   datetime.datetime(2020,7,19,5,1,3,222),
                   datetime.datetime(2020,7,21,0,0,0,0),
                   datetime.datetime(2020,7,21,0,0,0,0),
                   datetime.datetime(2020,7,21,3,2,1,110),
                   datetime.datetime(2020,7,23,0,0,0,0),
                   datetime.datetime(2020,7,24,0,0,0,0),
                   datetime.datetime(2020,7,25,0,0,0,0)]

    # y = incremental_average(coords)
    # print(y)

    from sklearn.cluster import AgglomerativeClustering
    # clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=1.0)
    clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=1.0)
    # print(sim_map)
    # print(dist_mat)
    y_pred = clustering.fit(coords[:4]).labels_
    print('orig batch:', y_pred)

    clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=1.0)
    # print(sim_map)
    # print(dist_mat)
    y_pred = clustering.fit(coords[4:]).labels_
    print('orig batch:', y_pred)

    clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=1.0)
    # print(sim_map)
    # print(dist_mat)
    y_pred = clustering.fit(coords).labels_
    print('orig batch:', y_pred)

    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import completeness_score
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import v_measure_score

    params = {'n_samples': 90,
            'CC_count': 11,
            'mean_min': -10.0,
            'mean_max': 10.0,
            'covar_min': 0.5,
            'covar_max': 0.9,
            'stationary_prob': 0.3,
            'random_state': 1235}
    # coords, y = synthetic_CC_hmm(hyper_params=params)
    # coords_normalized = preprocessing.normalize(coords, norm='l2')
    
    # y_label_inc = incremental_average(coords, hyper_params={'dist_threshold': 1.0})
    # print(y_label_inc)

    dist_mat = compute_dist_mat(np.array(coords))
    # print('dist mat: \n', dist_mat)
    sim_mat = sim_dist_convert(dist_mat)
    # print('sim mat: \n', sim_mat)
    sim_map = mat_to_dict(sim_mat, no_zeros=True)
    # print(sim_map)
    # print(dist_mat)

    batch_params = {'input_format': 'similarity_dict',
                    'required_format': 'dist_mat',
                    'distance_threshold': 1.0,
                    'max_sim_value': 2.01246118,
                    'n_clusters': None,
                    'data_size': None}

    keep_timestamp_results = {'active': True, 'v_scores': [], 'batch_params': batch_params, 'complete_batch': y_pred, 'v_scores_complete': [], 'size_list': [], 'h_scores': [], 'h_scores_complete': [], 'c_scores': [], 'c_scores_complete': []}
    inc_params = {'dist_threshold': 1.0,
            'data_size': 8,
            'time_stamps': time_stamps,
            'window_size': 4,
            'dist_mat': False,
            'max_sim_value': 2.01246118,
            'linkage': 'centroid',
            'keep_timestamp_results': keep_timestamp_results,
            'update_clusters': False}
    y_label_inc_SI = incremental_average_SI(sim_map, hyper_params=inc_params)
    print('complete incremental: ', y_label_inc_SI)
    print('final: ', keep_timestamp_results)
    exit(0)


    warm_params = {'batch_portion': 0.5,
              'data_size': 8,
              'dist_threshold': 1.0,
              'batch_threshold': 1.0,
              'time_stamps': None,
              'window_size': 15,
              'batch_params': {'max_sim_value': 2.01246118},
              'inc_params': inc_params}

    y_label_inc, y_label_batch = warm_start_inc_avg_SI(sim_map, hyper_params=warm_params)
    print('batch:', y_label_batch)
    print(y_label_inc)


    # for i in range(len(y_label_inc)):
    #     print(y_label_inc[i] == y_label_inc_SI[i])

    exit(0)
    

    # clustering = AgglomerativeClustering(distance_threshold=1.4, n_clusters=None, linkage='average').fit(coords)
    # y_label_batch = clustering.labels_


    inc_result = []
    inc_c = []
    inc_h = []
    batch_result = []
    batch_c = []
    batch_h = []

    x = [] 
    for dist in np.arange(0.05, 10.0, 0.1):
        tmp_inc_result = []
        tmp_inc_c = []
        tmp_inc_h = []
        tmp_batch_result = []
        tmp_batch_c = []
        tmp_batch_h = []

        for rs in [1, 12, 123, 1234, 12345, 2345, 345, 45, 5]:
            params = {'n_samples': 90,
                    'CC_count': 11,
                    'mean_min': -10.0,
                    'mean_max': 10.0,
                    'covar_min': 0.5,
                    'covar_max': 0.9,
                    'stationary_prob': 0.3,
                    'random_state': rs}
            coords, y = synthetic_CC_hmm(hyper_params=params)
            coords_normalized = preprocessing.normalize(coords, norm='l2')

            y_label_inc = incremental_average(coords, hyper_params={'dist_threshold': dist})

            clustering = AgglomerativeClustering(distance_threshold=dist, n_clusters=None, linkage='average').fit(coords)
            y_label_batch = clustering.labels_
        
            tmp_inc_result.append(v_measure_score(y, y_label_inc))
            tmp_inc_c.append(completeness_score(y, y_label_inc))
            tmp_inc_h.append(homogeneity_score(y, y_label_inc))

            tmp_batch_result.append(v_measure_score(y, y_label_batch))
            tmp_batch_c.append(completeness_score(y, y_label_batch))
            tmp_batch_h.append(homogeneity_score(y, y_label_batch))

        x.append(dist)
        inc_result.append(np.mean(tmp_inc_result))
        inc_c.append(np.mean(tmp_inc_c))
        inc_h.append(np.mean(tmp_inc_h))

        batch_result.append(np.mean(tmp_batch_result))
        batch_c.append(np.mean(tmp_batch_c))
        batch_h.append(np.mean(tmp_batch_h))
            
    print('batsh:', np.max(batch_result))
    print('inc:', np.max(inc_result))

    plt.plot(x, inc_result, label='inc V')
    plt.plot(x, inc_c, label='inc C')
    plt.plot(x, inc_h, label='inc H')
    plt.plot(x, batch_result, label='batch V')
    plt.plot(x, batch_c, label='batch C')
    plt.plot(x, batch_h, label='batch H')
    plt.legend()
    plt.savefig('comparison_ACU.png')

    plt.clf()

    """
    y_label_inc = incremental_average(coords, hyper_params={'dist_threshold': 1.40})
    viz.viz_2d(coords, y_label_inc, 'inc_out.png', y=y)
    plt.clf()

    clustering = AgglomerativeClustering(distance_threshold=0.45, n_clusters=None, linkage='average').fit(coords)
    y_label_batch = clustering.labels_
    viz.viz_2d(coords, y_label_batch, 'batch_out.png', y=y)
    """

