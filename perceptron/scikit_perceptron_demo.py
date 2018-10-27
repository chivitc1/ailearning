from sklearn import datasets
import numpy as np


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class lebels: ', np.unique(y))
print(iris.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

print('Number of test samples', X_test.shape[0])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron

model = Perceptron(max_iter=40, eta0=0.1, random_state=1)
model.fit(X_train_std, y_train)

y_pred = model.predict(X_test_std)
print('Misclassified samples: {} '.format((y_test != y_pred).sum()))

from sklearn.metrics import accuracy_score

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
