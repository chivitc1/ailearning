# Perceptron model

Output must be a linear function of Input features

## Scikit perceptron demo

### Prepare data
Use Iris dataset from scikit

Select feature use for training: for simple demo we use only 2 of 4 Iris features

For performance, we use labels/target as numeric values instead of string labels

As supervised learning, we need to split input data into training set and test set
using scikit train_test_split as convenience

train_test_split() random_state: pass to assure the same random generation for demonstration only.

train_test_split() stratify: = y => to assure the rate of each label is balance to avoid overfitting, underfitting

Numpy bincount(array_val): count appearance of each label


#### Scaling training data
Use StandardScaler: for feature scaling. Some Algorithm require this for optimization.

Using the fit method, StandardScaler estimated the
parameters μ (sample mean) and σ (standard deviation) for each feature dimension
from the training data

By calling the transform method, we then standardized the
training data using those estimated parameters µ and σ .

Note that we used the
same scaling parameters to standardize the test set so that both the values in the
training and test dataset are comparable to each other

### Create model from scikit Perceptron
model = Perceptron(n_tier=40, eta0=0.1, random_state=1)

n_iter is epoch

eta0 is learning rate

Finding an appropriate learning rate requires some experimentation.
If the learning rate is too large, the algorithm will overshoot the global cost
minimum. If the learning rate is too small, the algorithm requires more epochs until
convergence, which can make the learning slow—especially for large datasets.

We used the random_state parameter to ensure the reproducibility of the initial
shuffling of the training dataset after each epoch.

### Train model

model.fit(X_train_std, y_train)

### Evaluate model accuracy

accu = accuracy_score(y_test, y_pred)

