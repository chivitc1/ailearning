# Support Vector Machine (SVM)

Can be considered an extension of the perceptron

The perceptron algorithm minimized misclassification errors

SVMs optimization objective is to maximize the margin

The margin is defined as the
distance between the separating hyperplane (decision boundary) and the training
samples that are closest to this hyperplane, which are the so-called support vectors.

The rationale behind having decision boundaries with large margins is that they
tend to have a lower generalization error
whereas models with small margins are
more prone to overfitting

All negative samples should fall on one side
of the negative hyperplane, whereas all the positive samples should fall behind the
positive hyperplane

## alternative SVM

sometimes our datasets are too large to fit into computer memory

scikit-learn also offers alternative implementations via
the SGDClassifier class, which also supports online learning via the partial_fit method.


