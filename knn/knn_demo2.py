import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def main():
    """shows the five points that are closest to the test data point"""
    # Define sample 2D datapoints
    X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9],
                  [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9],
                  [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

    # Define the number of nearest neighbors you want to extract
    k = 5

    # Define a test datapoint that will be used to extract the K nearest neighbors
    test_datapoint = [4.3, 2.7]

    # Plot the input data using circular shaped black markers
    plt.figure()
    plt.title('Input data')
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='black')

    # Build K Nearest Neighbors model
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)

    # extract the nearest neighbors to our test data point
    distances, indices = knn_model.kneighbors([test_datapoint])

    # Print the 'k' nearest neighbors
    print("\nK Nearest Neighbors:")
    for rank, index in enumerate(indices[0][:k], start=1):
        print(str(rank) + " ==>", X[index])

    # Visualize the nearest neighbors along with the test datapoint
    plt.figure()
    plt.title('Nearest neighbors')
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='k')
    plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1],
    marker='o', s=250, color='k', facecolors='none')
    plt.scatter(test_datapoint[0], test_datapoint[1],
    marker='x', s=75, color='k')

    plt.show()

if __name__ == '__main__':
    main()