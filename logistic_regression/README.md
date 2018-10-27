# Logistic regression model

A model for classification

To solve linear and binary classification problems

Can be extended to multiclass classification (via the OvR technique)

Performs very well on linearly separable classes

One of the most widely used algorithms for classification in industry

A Probabilistic model

## scikit LogisticRegression model

Support multiclasses classification by default

Optimized

### Create model

model = LogisticRegression(C=100.0, random_state=1)

C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

###  Class-membership probabilities

The probability that training examples belong to a certain class can be computed
using the predict_proba method

## Predict

y_pred = model.predict(X_test_std)
