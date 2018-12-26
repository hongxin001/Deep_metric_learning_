import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from utils import to_numpy

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def NMI(X, ground_truth, n_cluster=3):
    X = [to_numpy(x) for x in X]
    X = np.array(X)
    ground_truth = np.array(ground_truth)

    kmeans = KMeans(n_clusters=n_cluster, n_jobs=-1, random_state=0).fit(X)

    print('K-means done')
    nmi = normalized_mutual_info_score(ground_truth, kmeans.labels_, average_method="arithmetic")
    return nmi


def main():
    label = [1, 2, 3]*2

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])

    print(NMI(X, label))

if __name__ == '__main__':
    main()