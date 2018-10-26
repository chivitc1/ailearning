import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


# loading training data
iris = datasets.load_iris()


def explore_iris_dataset():
    print(type(iris.feature_names))
    print(iris.feature_names.__len__())
    print(iris.feature_names)

    print(type(iris.target_names))
    print(iris.target_names.shape)

    print(type(iris.data))
    print(iris.data.shape)

    print(type(iris.target))
    print(iris.target.shape)

    print(iris.data[:, :2].shape)
    print(iris.data[:, 0].shape)
    print(iris.data[:, -1].shape)

    print(iris.data[:, 0].min())
    print(iris.data[:, 0].max())

    print(iris.feature_names[0:2])
    X = iris.data[:, [0, 1]]
    y = iris.target
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.xticks(())
    plt.yticks(())

    plt.show()


def demo_knn():
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('There are {} samples in the training set and {} samples in the test set'.format(
        X_train.shape[0], X_test.shape[0]))
    sc = StandardScaler()

    #Compute the mean and std to be used for later scaling
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y_test))])
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    c=cmap(idx), marker=markers[idx], label=cl)
    # plt.show()

    """
    KNN distance metrics: euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis
    """
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn_model.fit(X_train_std, y_train)
    print('The accuracy of the knn classifier is {:.2f} on test data'.format(knn_model.score(X_test_std, y_test)))

    print('Input data: {}'.format(X_test_std[44]))
    result = knn_model.predict([X_test_std[44]])
    print('Predict result: {}'.format(iris.target_names[result[0]]))


def demo_knn_no_std():
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y_test))])
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    c=cmap(idx), marker=markers[idx], label=cl)
    # plt.show()

    """
    KNN distance metrics: euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis
    """
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn_model.fit(X_train, y_train)
    print('The accuracy of the knn classifier is {:.2f} on test data'.format(knn_model.score(X_test, y_test)))

    print('Input data: {}'.format(X_test[44]))
    result = knn_model.predict([X_test[44]])
    print('Predict result: {}'.format(iris.target_names[result[0]]))


if __name__ == '__main__':
    # explore_iris_dataset()
    # demo_knn()
    demo_knn_no_std()
