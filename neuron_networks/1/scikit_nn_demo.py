from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

cancer = load_breast_cancer()
print('Num of samples', cancer.data.shape[0])
X = cancer.data
y = cancer.target
print('Num of features', cancer.data.shape[1])
print('Feature names:')
print(cancer.feature_names)

print('Classes')
print(cancer.target_names)

print('First sample:')
print(X[:1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

def nn_test():
    model.add(Dense(15, input_dim=30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=50)
    y_pred = model.predict_classes(X_test)
    print('First 5 test samples prediction: ', y_pred[:5])
    accu = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accu)


def deep_nn_test():
    model.add(Dense(15, input_dim=30, activation='relu'))

    # Add hidden layers to be deep learning
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=50)
    y_pred = model.predict_classes(X_test)
    print('First 5 test samples prediction: ', y_pred[:5])
    accu = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accu)


if __name__ == '__main__':
    # nn_test()
    deep_nn_test()
