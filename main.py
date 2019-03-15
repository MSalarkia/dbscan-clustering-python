import numpy as np
import matplotlib.pyplot as plt
from dbscan import dbscan


def main(epsilon=1, min_points=5):
    # Load Data
    n1 = 500
    data1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n1)

    n2 = 1000
    r = np.random.uniform(6, 12, n2)
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, n2)
    data2 = np.array([r * np.cos(theta), r * np.sin(theta)]).T

    X = np.vstack((data1, data2))

    # Run dbscan Clustering Algorithm
    indices, is_noise = dbscan(X, epsilon, min_points)

    # Plot Results
    plt.title(r'dbscan Clustering ($\epsilon$ = ' + str(epsilon) + ', min_points = ' + str(min_points) + ')')
    # plot_clustering_results(X, indices)
    plt.scatter(X[np.logical_not(is_noise), 0], X[np.logical_not(is_noise), 1], c=indices[np.logical_not(is_noise)])
    plt.plot(X[np.bool8(is_noise), 0], X[np.bool8(is_noise), 1], '*g', markersize=10)
    plt.show(block=False)
    return indices, is_noise


def plot_clustering_results(X, indices):
    k = int(max(indices))
    legends = []

    for i in range(k + 1):
        Xi = X[indices == i, :]
        if i != 0:
            style = 'x'
            marker_size = 8
            legends.append('Cluster #' + str(i))
        else:
            style = 'o'
            marker_size = 6
            if len(Xi) != 0:
                legends.append('Noise')

        if len(Xi) != 0:
            plt.plot(Xi[:, 0], Xi[:, 1], style, markersize=marker_size)

        plt.hold()

    plt.axis('equal')
    plt.grid()
    plt.legend(legends)
    plt.legend(loc='upper right')

    plt.show(block=False)


if __name__ == "__main__":
    main()
