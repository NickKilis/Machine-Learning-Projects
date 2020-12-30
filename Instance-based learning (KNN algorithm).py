'''
Instance-based learning (K-NEAREST NEIGHBORS algorithm) 

From sklearn, we will import:
    'model_selection'  : package, which will help validate our results
    'metrics' package  : for measuring scores
    'preprocessing'    : package, for rescaling ('normalizing') our data
    'neighbors'        : package, for creating and using KNN classfier

We also need 'pandas' and 'numpy' libraries for manipulating data.
'matplotlib.pyplot' is ncessary as well, for plotting results.
'''
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import data from file. Pandas will read this file as a 'dataframe' object (using 'read_csv' command)
# so we will treat that object accordingly.
# (FYI, a dataframe object has more information than just values, but we'll
# stick just to values for this project)
diabetesData =pd.read_csv('diabetes.csv') 

# =============================================================================
# From diabetesData object, read included features and target variable.
# You can select a specific row/column of the dataframe object by using 'iloc',
# as if you were accessing an array (tip: you can access the last row or column
# with 'iloc' using the '-1' index).
# Make sure that, after selecting (/accessing) the desired rows and columns,
# you extract their 'values', which is what you want to store in variables 'X' and 'y'.
rows = diabetesData.shape[0]
cols = diabetesData.shape[1]
X = diabetesData.iloc[0:rows , 0:cols-1]
y = diabetesData.iloc[0:rows, cols-1:cols]

# =============================================================================
# KNN is sensitive to the magnitudes of the features' values, so we must rescale
# data to the [0, 1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
minMaxScaler = MinMaxScaler()
X_rescaled = minMaxScaler.fit_transform(X)

# =============================================================================
# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# The 'stratify' parameter will split the dataset proportionally to the target variable's values.
# Also, 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = train_test_split(X_rescaled,y,stratify=y,random_state=0)

# =============================================================================
# Initialize variables for number of neighbors, and arrays for holding values
# for recall, precision and F1 scores.
k = 200
neighbors = np.arange(1, k+1)
parameter1 = 'macro'
weights = ['distance', 'uniform']
p_values = [1,2,4]
#training_accuracy = []
#test_accuracy = []
for w in weights:
    for p_value in p_values:
        best_F1 = None
        bestF1_ind = None
        prec = np.arange(1, k+1, dtype = np.float64)
        rec = np.arange(1, k+1, dtype = np.float64)
        f1 = np.arange(1, k+1, dtype = np.float64)
        
        # Run the classification algorithm 'k' times (k = number of neighbors)
        for n in range(1,k+1):
            # NeighborsClassifier is the core of this script. You can customize its functionality
            # in various ways- for this project, just tweak the following parameters:
            # 'n_neighbors': The number of neighbors to use.
            # 'weights': Can be either 'uniform' (i.e. all points have equal weights) or 'distance' (points are
            #            assigned weights according to their distance from the query point).
            # 'metric': The metric used for measuring the distance between points. Could 'euclidean',
            #           'manhattan', 'minkowski', etc. Keep in mind that you need to tweak the 
            #           power parameter 'p' as well when using the 'minkowski' distance.
            model = KNeighborsClassifier(n_neighbors=n,weights=w,metric='minkowski',p=p_value)
            
            # =============================================================================
            # Let's train our model.
            model.fit(x_train, y_train.values.ravel())
            
            # =============================================================================
            # Ok, now let's predict the output for the second subset
            y_predicted = model.predict(x_test)
            
            # =============================================================================
            # Time to measure scores. We will compare predicted output (from input of x_test)
            # with the true output (i.e. y_test).
            # You can call 'recall_score()', 'precision_score()', 'f1_score()' or any other available metric
            # from the 'metrics' library.
            # The 'average' parameter is used while measuring metric scores to perform 
            # a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET. USE EITHER 
            # 'MICRO' OR 'MACRO' (preferably 'macro' for final results).
            
            # =============================================================================
            result = precision_recall_fscore_support(y_test, y_predicted, average='macro')
            #training_accuracy.append(accuracy_score(x_train, y_train.astype(float),normalize=False))
            #test_accuracy.append(accuracy_score(x_test.round(), y_test.astype(int),normalize=False))
            prec[n-1] = result[0]
            rec[n-1] = result[1]
            f1_score = result[2]
            f1[n-1] = f1_score
                
            # Get best value of F1 score, and its index
            if best_F1 is None and bestF1_ind is None:
                best_F1 = f1_score
                bestF1_ind = n
            else:
                if best_F1 < f1_score:
                    best_F1 = f1_score
                    bestF1_ind = n

            print('# =================================================================== #')
            print('Weight:                             ' + str(w))
            print('Minkowski power:                    ' + str(p_value))
            print('Best F1:                            ' + str(best_F1))
            print('Number of neighbors with best F1:   ' + str(bestF1_ind))
            print('Precision:                          ' + str(prec[bestF1_ind - 1]))
            print('Recall:                             ' + str(rec[bestF1_ind - 1]))
            print('Average:                            ' + str(parameter1))
            print('# =================================================================== #')
              
# Plot stored results
df = pd.DataFrame({'Number of Neighbors': neighbors, 'Precision': prec, 'Recall': rec, 'F1': f1 })

# Plot data
plt.plot('Number of Neighbors', 'Precision', data=df, color='red')
plt.plot('Number of Neighbors', 'Recall', data=df, color='blue')
plt.plot('Number of Neighbors', 'F1', data=df, color='green')
plt.legend()
plt.title(label='KNN algorithm, weight = ' + str(w) + ', Minkowski power = ' + str(p_value))
plt.xlabel('Number of Neighbors')
plt.ylabel('Metrics')
plt.savefig('results/KNN_diabetes_plot_' + str(w) + '_' + str(p_value) + '.png')
plt.clf()

# =============================================================================
# Calculate accuracy for the optimal number of F1
def calc_accuracy(N_neighbors,weight,P):
    model2 = KNeighborsClassifier(n_neighbors=N_neighbors,weights=weight,metric='minkowski',p=P)
    model2.fit(x_train, y_train.values.ravel())
    y_pred = model2.predict(x_test)            
    print(accuracy_score(y_test, y_pred))
    
case=1    
if case==1:
    N_neighbors=3
    weight='uniform'           
    P=2
    calc_accuracy(N_neighbors,weight,P)
elif case==2:
    N_neighbors=18
    weight='distance'           
    P=2
    calc_accuracy(N_neighbors,weight,P)
elif case==3:
    N_neighbors=11
    weight='uniform'           
    P=1
    calc_accuracy(N_neighbors,weight,P)
elif case==4:
    N_neighbors=14
    weight='distance'           
    P=1
    calc_accuracy(N_neighbors,weight,P)
elif case==5:
    N_neighbors=3
    weight='uniform'           
    P=4
    calc_accuracy(N_neighbors,weight,P)
elif case==6:
    N_neighbors=13
    weight='distance'           
    P=4
    calc_accuracy(N_neighbors,weight,P) 