"""This module contains various evaluation methods."""
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import completeness_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import f1_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import homogeneity_completeness_v_measure 
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score


def purity(labels_true, labels_pred):
    """Computes the purity score

    Args:
        labels_true: The true labels of the elements.
        labels_pred: The predictted labels of the elements.

    Returns:
        The purity score.
    """
    aggregate = 0
    for cluster_id in range(max(labels_pred)+1):
        including_classes = np.array(labels_true)[labels_pred == cluster_id]
        if len(including_classes) > 0:
            aggregate += max(list(np.bincount(including_classes)))
    purity = aggregate / float(len(labels_true))
    return purity


def rand_index_score(labels_true, labels_pred):
    """Computes the Rand Index score
    
    Args:
        labels_true: The true labels of the elements.
        labels_pred: The predictted labels of the elements.

    Returns:
        The Rand Index score.
    """
    data_size = len(labels_true)
    TP = 0
    TN = 0
    for i in range(data_size):
        for j in range(i):
            if labels_true[i] == labels_true[j] and labels_pred[i] == labels_pred[j]:
                TP += 1

            if labels_true[i] != labels_true[j] and labels_pred[i] != labels_pred[j]:
                TN += 1

    total_pairs = data_size*(data_size-1) / 2
    RI = (TP + TN) / total_pairs
    return RI


def F_beta_score(labels_true, labels_pred, beta=1.0, get_prec_rec=False):
    """Computes the F score value with arbitrary beta. 
    
    Args:
        labels_true: The true labels of the elements.
        labels_pred: The predictted labels of the elements.
        beta: The beta value used in the formula of F-measure. A higher value
            emphesizes on Recall and a lower one accents the Precision.

    Returns:
        The F score, precision, and recall
    """
    data_size = len(labels_true)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(data_size):
        for j in range(i):
            if labels_true[i] == labels_true[j]:
                if labels_pred[i] == labels_pred[j]:
                    TP += 1
                else:
                    FN += 1

            if labels_true[i] != labels_true[j]:
                if labels_pred[i] != labels_pred[j]:
                    TN += 1
                else:
                    FP += 1

    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)

    if precision == 0 or recall == 0:
        return 0

    F_beta = ((beta**2 + 1) * precision * recall) / (beta**2 * precision + recall)

    if get_prec_rec:
        return F_beta, precision, recall
    return F_beta


def evaluate_results(y_true, y_pred, measure_list=[('ARI', adjusted_rand_score), ('AMI', adjusted_mutual_info_score)]):
    results_dict = {}
    for name, measure in measure_list:
        results_dict[name] = measure(y_true, y_pred)

    return results_dict


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    labels_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    labels_pred = np.array([1, 1, 1, 1, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 2, 2, 2])
    print(purity(labels_true, labels_pred))

    evaluation_methods = [('Homogeneity', homogeneity_score), ('Completeness', completeness_score),
            ('Nomalized MI', normalized_mutual_info_score),
            ('Adjusted MI', adjusted_mutual_info_score)]

    n_samples = 2000
    
    seed = 1234
    random_labels_true = np.random.RandomState(seed).randint
    labels_true = random_labels_true(low=0, high=11, size=n_samples)
    
    all_scores = {}
    for name, method in evaluation_methods:
        all_scores[name] = []

    x_values = []
    for n_clusters in range(5, 100, 5):
        x_values.append(n_clusters)
        random_labels_pred = np.random.RandomState(seed + n_clusters).randint
        labels_pred = random_labels_pred(low=0, high=n_clusters, size=n_samples)
        # labels_true = random_labels_true(low=0, high=n_clusters, size=n_samples)
        for name, method in evaluation_methods:
            score = method(labels_true, labels_pred)
            all_scores[name].append(score)

    print(all_scores)


    fig, ax = plt.subplots()
    
    for name in all_scores:
        ax.plot(x_values, all_scores[name], label=name)
        
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    legend_without_duplicate_labels(ax)
    plt.savefig('completeness_homogeneity_' + str(n_samples) + '_var.png')

