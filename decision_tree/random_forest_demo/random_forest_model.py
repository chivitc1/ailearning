from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(criterion='gini',
                               n_estimators=25,
                               random_state=1,
                               n_jobs=2)

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Number of train samples', X_train.shape[0])
print('Number of test samples', X_test.shape[0])


def random_forest_test():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
    accu = accuracy_score(y_test, y_pred)
    print(accu)


if __name__ == '__main__':
    random_forest_test()