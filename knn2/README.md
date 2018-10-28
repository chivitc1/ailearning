# K-nearest neighbors model

a lazy learner

it doesn't learn a discriminative function from the
training data, but memorizes the training dataset instead

## Parametric versus nonparametric models
Machine learning algorithms can be grouped into parametric and
nonparametric models.

Using parametric models, we estimate
parameters from the training dataset to learn a function that can
classify new data points without requiring the original training dataset
anymore.

Typical examples of parametric models are the perceptron,
logistic regression, and the linear SVM.

In contrast, nonparametric
models can't be characterized by a fixed set of parameters, and the
number of parameters grows with the training data.

Two examples of
non-parametric models are the decision tree classifier/random forest and the kernel SVM.

KNN belongs to a subcategory of nonparametric models that is
described as instance-based learning.

Models based on instance-based
learning are characterized by memorizing the training dataset, and lazy
learning is a special case of instance-based learning that is associated
with no (zero) cost during the learning process.

## KNN algorithm simple steps:

1. Choose the number of k and a distance metric.

2. Find the k-nearest neighbors of the sample that we want to classify.

3. Assign the class label by majority vote.

Based on the chosen distance metric, the KNN algorithm finds the k samples in the
training dataset that are closest (most similar) to the point that we want to classify.
The class label of the new data point is then determined by a majority vote among its
k nearest neighbors.

The main advantage of such a memory-based approach is that the classifier
immediately adapts as we collect new training data.

However, the downside is that
the computational complexity for classifying new samples grows linearly with the
number of samples in the training dataset

Furthermore, we can't discard training samples since no training step is involved. Thus, storage space
can become a challenge if we are working with large datasets.

## scikit KNN

