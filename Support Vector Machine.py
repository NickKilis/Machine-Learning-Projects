'''Support Vector Machine

From sklearn, we will import:
    'datasets'        : for loading data
    'model_selection' : package, which will help validate our results
    'metrics'         : package, for measuring scores
    'svm'             : package, for creating and using a Support Vector Machine classifier
    'preprocessing'   : package, for rescaling ('normalizing') our data

 We also need 'pandas' library to manipulate our data.
'''
# =============================================================================
import time
start = time.time()
from sklearn import metrics,svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# =============================================================================
# Import credit card dataset from file. 
# Pandas will read this file as a 'dataframe' object (using 'read_csv' command)
# so we will treat that object accordingly.
# (FYI, a dataframe object has more infromation than just values, but we'll
# stick just to values for this project).
myData = pd.read_csv('creditcard.csv') 
#myData=myData.iloc[0:50000-1,0:31]

# =============================================================================
# From myData object, read included features and target variable.
# You can select a specific row/column of the dataframe object by using 'iloc',
# as if you were accessing an array (tip: you can access the last row or column
# with 'iloc' using the '-1' index).
# Make sure that, after selecting (/accessing) the desired rows and columns,
# you extract their 'values', which is what you want to store in variables 'X' and 'y'.
rows = myData.shape[0]
cols = myData.shape[1]
X = myData.iloc[1:rows , 0:cols-1].values
y = myData.iloc[1:rows, cols-1].values 

# =============================================================================
# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

# =============================================================================
# Suport Vector Machines, like other classification algorithms, are sensitive to the magnitudes 
# of the features' values. Since we already split our dataset into 'train' and 'test' sets,
# we must rescale them separately (but with the same scaler).
# So, we rescale train data to the [0,1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
# Note that, test data should be transformed ONLY (not fit+transformed), since we require the exact
# same scaler used for the train data. 
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(x_train)
x_train = minMaxScaler.transform(x_train)
x_test = minMaxScaler.transform(x_test)

'''
Now we are ready to create our SVM classifier. Scikit-learn has more than one type of SVM classifier,
so, let us all agree on using the 'SVC' classifier (i.e. the fundamental model).
C:      This parameter, also called 'penalty', is used to control the 'decisiveness' of the SVM.
        In essence, it is used to guide the SVM when deciding between creating a smooth surface,
        i.e. larger margins in the hyperplane and thus more misclassifications (low C)
        or choosing smaller separation margins and thus lowering misclassification levels (high C).
        Usually, we go for a high C value (i.e. more correct classifications).
kernel: This is one of the most important parameters in a Support Vector Machine classifier.
        Simply put, it provides a way for the classifier to transform the current representation
        of the samples in the hyperplane, into another kind of representation (i.e.  create a mapping),
        where it is easier to separate the data.
        Available options for this parameters are 'linear', 'poly' (polynomial),
        'rbf' (Radial Basis Function) and 'sigmoid'.
degree: Only used when kernel = 'poly'. This parameter is used to define the
        degree of the polynomial.
gamma:  Only used when working with 'rbf', 'poly' and 'sigmoid' kernels. The effect 
        of this parameter resembles the effect of the number of neighbors in 
        K-Nearest Neighbors classification. A low 'gamma' value cannot produce great results
        because it cannot 'see' the underlying shape that can 'hug' (or, group together) 
        similar points in the hyperplane, while a high 'gamma' value is highly likely to
        overfit the model (and thus will not be able to generalize well due to high variance and low bias).
        Scikit-learn can choose a good value for gamma (passing 'auto' uses 1/n_features 
        as a gamma value, although it will be replaced by something similar in the next version).

Note that a good model can be produced by keeping a good balance between 'C' and 'gamma'!
'''
def predict(model, x_test, y_test, kernel, C=1.0, gamma='-', degree='-', results=None):
    y_predicted = model.predict(x_test)
    confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
    result = metrics.precision_recall_fscore_support(y_test, y_predicted, average='macro')
    accuracy=metrics.accuracy_score(y_test, y_predicted)
    results = results.append({'C': C,
                              'Kernel': kernel, 
                              'Gamma': gamma,
                              'Degree': degree,
                              'Accuracy': float("%0.2f"%accuracy),
                              'Recall': float("%0.2f"%result[1]),
                              'Precision':  float("%0.2f"%result[0]),
                              'F1': float("%0.2f"%result[2]),
                              'Normal': confusionMatrix[0],
                              'Fraud': confusionMatrix[1]}, ignore_index=True)
    return (results)


kernels=['poly', 'rbf', 'sigmoid']
C_sig = [0.1, 10,100]
C_p_rbf = [0.1, 10]
#gammas = [ 0.2, 0.3, 0.5, 2, 5, 6]
gammas_p = [ 0.2, 6]
gammas_rbf = [ 0.3, 5]
gammas_sig = [ 0.5,2, 5]
degrees_p = [2, 5]

results = pd.DataFrame(columns=[ 'C','Kernel', 'Gamma', 'Degree','Accuracy', 'Recall','Precision', 'F1', 'Normal', 'Fraud'])

for kernel in kernels:
            if kernel == 'poly':
                for c in C_p_rbf:
                    for degree in degrees_p:
                        for gamma in gammas_p:
                            model = svm.SVC(kernel=kernel, C=c, degree=degree, gamma=gamma)
                            model.fit(x_train, y_train)
                            results = predict(model=model, x_test=x_test, y_test=y_test, kernel=kernel, C=c, gamma=gamma, degree=degree, results=results)
            elif kernel == 'rbf':
                for c in C_p_rbf:
                    for gamma in gammas_rbf:
                        model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
                        model.fit(x_train, y_train)
                        results = predict(model=model, x_test=x_test, y_test=y_test, kernel=kernel, C=c, gamma=gamma, results=results)
            elif kernel == 'sigmoid':
                for c in C_sig:
                    for gamma in gammas_sig:
                        model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
                        model.fit(x_train, y_train)
                        results = predict(model=model, x_test=x_test, y_test=y_test, kernel=kernel, C=c, gamma=gamma, results=results)
results.to_csv('results.csv')

end = time.time()
elapsed = end - start #7810.381619215012