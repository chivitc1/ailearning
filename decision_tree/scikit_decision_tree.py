from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Number of train samples', X_train.shape[0])
print('Number of test samples', X_test.shape[0])


def tree_test():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)

    accu = accuracy_score(y_test, y_pred)
    print(accu)
    plot_tree(model)


def plot_tree(_tree_model):
    from pydotplus import graph_from_dot_data
    from sklearn.tree import export_graphviz
    dot_data = export_graphviz(_tree_model,
                               filled=True,
                               rounded=True,
                               class_names=['Setosa', 'Versicolor', 'Virginica'],
                               feature_names=['petal length', 'petal width'],
                               out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('tree.png')

if __name__ == '__main__':
    tree_test()