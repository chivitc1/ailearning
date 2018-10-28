# Decision tree model

Decision trees can build complex decision boundaries by dividing the feature
space into rectangles

However, we have to be careful since the deeper the decision
tree, the more complex the decision boundary becomes, which can easily result in
overfitting

Although feature scaling may
be desired for visualization purposes, note that feature scaling is not a requirement
for decision tree algorithms

Gini impurity is a measure of misclassification, which applies
in a multiclass classifier context


## Visualize tree

Install Graphviz executable into your system:

$ sudo apt-get install graphviz

So we can generate tree png image for visualize

## Explain tree

- We started with
105 samples at the root and split them into two child nodes with 35 and 70 samples,
using the petal width cut-off ≤ 0.75 cm

- After the first split, we can see that the left
child node is already pure and only contains samples from the Iris-setosa class
(Gini Impurity = 0)

- The further splits on the right are then used to separate the
samples from the Iris-versicolor and Iris-virginica class.
