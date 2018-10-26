# Problem:

Movie recommendation system

# Type:

- Classification, multi classes

- Non parametric

Non-parametric means there is no assumption for underlying data distribution.
In other words, the model structure determined from the dataset.
This will be very helpful in practice where most of the real world datasets
do not follow mathematical theoretical assumptions

- Instance-based learning algorithm: our algorithm doesn’t explicitly learn a model.
Instead, it chooses to memorize the training instances which are subsequently used as
“knowledge” for the prediction phase.

Only when a query to our database is made (i.e. when we ask it to predict a label given
an input), will the algorithm use the training instances to spit out an answer

The minimal training phase of KNN comes both at a memory cost, since we must store
a potentially huge data set, as well as a computational cost during test time since
classifying a given observation requires a run down of the whole data set

# Algorithm summary:

A K-Nearest Neighbors classifier is a classification model that uses the nearest neighbors
algorithm to classify a given data point. The algorithm finds the K closest data points in the
training dataset to identify the category of the input data point. It will then assign a class to
this data point based on a majority vote

From the list of those K data points, we look at the
corresponding classes and pick the one with the highest number of votes

K is the number of nearest neighbors

K is generally an odd number if the number of classes is 2

The value of K depends on the problem at hand

# Basic steps:

1. Calculate distance
2. Find closest neighbors
3. Vote for labels

# Curse of Dimensionality

KNN performs better with a lower number of features than a large number of features. You can say that when the number of features increases than it requires more data. Increase in dimension also leads to the problem of overfitting. To avoid overfitting, the needed data will need to grow exponentially as you increase the number of dimensions. This problem of higher dimension is known as the Curse of Dimensionality.

To deal with the problem of the curse of dimensionality, you need to perform principal component analysis before applying any machine learning algorithm, or you can also use feature selection approach. Research has shown that in large dimension Euclidean distance is not useful anymore. Therefore, you can prefer other measures such as cosine similarity, which get decidedly less affected by high dimension.

# Computing similarity scores

In order to build a recommendation system, it is important to understand how to compare
various objects in our dataset

How to compare any two people with each other

## Euclidean score:

Euclidean distance between two objects is large,
then the Euclidean score should be low because a low score indicates that the objects are not
similar

score can range from 0 to 1

## Pearson score:

uses the covariance
between the two objects along with their individual standard deviations to compute the
score

score can range from -1 to +1

A score of +1 indicates that the objects are very similar

where a score of -1 would indicate that the objects are very dissimilar.

A score of 0 would indicate that there is no correlation between the two objects

## Finding similar users using collaborative filtering

The assumption here is that if two people have similar ratings for a particular set of movies,
then their choices in a set of new unknown movies would be similar too

By identifying patterns in those common movies, we make predictions about new movies.

We compare different users in the dataset using score techniques

This also works for finance, online shopping, marketing, customer studies, and so on.

## demo

cd ailearning/

python knn/movie_recommendation.py --user "Julie Hammel"
