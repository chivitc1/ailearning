from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

ppn_model = SGDClassifier(loss='perceptron')
lr_model = SGDClassifier(loss='log')
svm_model = SGDClassifier(loss='hinge')

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Number of train samples', X_train.shape[0])
print('Number of test samples', X_test.shape[0])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#########
def ppn_test():
    ppn_model.fit(X_train_std, y_train)

    y_pred = ppn_model.predict(X_test_std)
    print(y_pred)
    accu = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accu)

def logistic_regression_test():
    lr_model.fit(X_train_std, y_train)

    y_pred = lr_model.predict(X_test_std)
    print(y_pred)
    accu = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accu)

def svm_test():
    svm_model.fit(X_train_std, y_train)
    y_pred = svm_model.predict(X_test_std)
    print(y_pred)
    accu = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accu)


if __name__ == '__main__':
    # ppn_test()
    # logistic_regression_test()
    svm_test()