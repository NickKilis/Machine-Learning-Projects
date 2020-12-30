'''
LINEAR REGRESSION ALGORITHM

From 'sklearn' library, we need to import:
    'datasets'    : for loading our data
    'metrics'     : for measuring scores
    'linear_model': which includes the LinearRegression() method
From 'scipy' library, we need to import:
    'stats'       : which includes the spearmanr() and pearsonr() methods for computing correlation
Additionally, we need to import :
    'pyplot'      : from package 'matplotlib' for our visualization purposes
    'numpy'       : which implements a wide variety of operations
'''
# =============================================================================
# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
# Load diabetes data from 'datasets' class
diabetes = datasets.load_breast_cancer()

# =============================================================================
# Get samples from the data, and keep only the features that you wish.
# Load just 1 feature for simplicity and visualization purposes...
# X: features
# Y: target value (prediction target)
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# =============================================================================
# Create linear regression model.
linearRegressionModel = linear_model.LinearRegression()

# =============================================================================
# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y)

# =============================================================================
# Let's train our model.
linearRegressionModel.fit(x_train,y_train)

# =============================================================================
# Ok, now let's predict the output for the test input set
y_predicted = linearRegressionModel.predict(x_test)

# =============================================================================
# Time to measure scores. We will compare predicted output (resulting from input x_test) with the true output (i.e. y_test).
# You can call 'pearsonr()' or 'spearmanr()' methods for computing correlation,
# 'mean_squared_error()' for computing MSE,
# 'r2_score()' for computing r^2 coefficient.

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Correlation via Pearsonr: %.3f , %.3f" % pearsonr(y_test,y_predicted))
print("Mean squared error (MSE): %.3f"        % mean_squared_error(y_test, y_predicted))
print("R^2 : %.3f"                            % r2_score(y_test,y_predicted))

# =============================================================================
# Plot results in a 2D plot (scatter() plot, line plot())
plt.figure(1, figsize=(9, 6))
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_predicted, color='blue', linewidth=3)
# Display 'ticks' in x-axis and y-axis
plt.xticks()
plt.yticks()
# Show plot
plt.show()
