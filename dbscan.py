import numpy as np
from scipy.spatial.distance import cdist


def dbscan(X, epsilon, min_points):
    def expand_cluster(i, neighbors, C):
        indices[i] = C

        k = 0
        while True:
            j = neighbors[k]

            if not visited[j]:
                visited[j] = 1
                neighbors2 = region_query(j)
                if len(neighbors2) >= min_points:
                    neighbors = np.append(neighbors, neighbors2)

            if indices[j] == 0:
                indices[j] = C

            k += 1
            if k + 1 > len(neighbors):
                break

    def region_query(i):
        return np.where(D[i, :] <= epsilon)[0]

    C = 0

    n = X.shape[0]
    indices = np.zeros((n,))

    D = cdist(X, X)

    visited = np.zeros((n,))
    is_noise = np.zeros((n,))

    for i in range(n):
        if not visited[i]:
            visited[i] = 1

            neighbors = region_query(i)
            if len(neighbors) < min_points:
                # X(i,:) is NOISE
                is_noise[i] = 1
            else:
                C = C + 1
                expand_cluster(i, neighbors, C)

    return indices, is_noise
