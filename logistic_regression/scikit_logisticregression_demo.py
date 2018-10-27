from sklearn import datasets
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=100.0, random_state=1)

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

model.fit(X_train_std, y_train)

# predict the probabilities of the first three samples in the test set:
print('Probability of first 3 test samples: ')
print(model.predict_proba(X_test_std[:3, :]).round(6))
print(model.predict_proba(X_test_std[:3, :]).sum(axis=1))

print('Predict class label by identify the largest column in each row:')
print(model.predict_proba(X_test_std[:3, :]).argmax(axis=1))

print(iris.target_names)

# Direct predict
y_pred = model.predict(X_test_std)
print(y_pred)

from sklearn.metrics import accuracy_score

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
