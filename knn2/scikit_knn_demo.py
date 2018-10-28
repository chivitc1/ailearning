from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Number of train samples', X_train.shape[0])
print('Number of test samples', X_test.shape[0])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


def knn_test():
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    print(y_pred)
    accu = accuracy_score(y_test, y_pred)
    print(accu)


if __name__ == '__main__':
    knn_test()
