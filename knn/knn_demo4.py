import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn import datasets


def train(data):

    # Load dataset
    wine = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# define column names
names = ['']
# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4])
y = np.array(df['class'])

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

if __name__ == '__main__':
    rating_file = 'knn/ratings.json'