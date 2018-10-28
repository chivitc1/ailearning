# Random forest model

Random forests have gained huge popularity in applications of machine learning
during the last decade due to their good classification performance, scalability,
and ease of use

a random forest can be considered as a combination of decision trees

The idea behind a random forest is to average multiple (deep)
decision trees that individually suffer from high variance, to build a more robust
model that has a better generalization performance and is less susceptible to
overfitting

Simple steps:

1. Draw a random bootstrap sample of size n (randomly choose n samples from
the training set with replacement).

2. Grow a decision tree from the bootstrap sample. At each node:

a. Randomly select d features without replacement.

b. Split the node using the feature that provides the best split according
to the objective function, for instance, maximizing the information gain.

3. Repeat the steps 1-2 k times.

4. Aggregate the prediction by each tree to assign the class label by majority
vote

The only parameter that we really need to care about in practice is the
number of trees k (step 3) that we choose for the random forest. Typically, the larger
the number of trees, the better the performance of the random forest classifier at the
expense of an increased computational cost.

## scikit randomforest implementation

In most implementations, including the RandomForestClassifier implementation
in scikit-learn, the size of the bootstrap sample is chosen to be equal to the number
of samples in the original training set, which usually provides a good bias-variance
tradeoff.

For the number of features d at each split, we want to choose a value that is
smaller than the total number of features in the training set. A reasonable default that
is used in scikit-learn and other implementations is d = sqrt(m) , where m is the number
of features in the training set.

