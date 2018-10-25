# Problem:

Movie recommendation system

# Type:

classification, multi classes

# Algorithm summary:

A K-Nearest Neighbors classifier is a classification model that uses the nearest neighbors
algorithm to classify a given data point. The algorithm finds the K closest data points in the
training dataset to identify the category of the input data point. It will then assign a class to
this data point based on a majority vote

From the list of those K data points, we look at the
corresponding classes and pick the one with the highest number of votes

The value of K depends on the problem at hand

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
