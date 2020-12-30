'''
Model evaluation (LOOCV - Friedman test)

PART 1
From any dataset that you like, train a machine learning model and calculate
the accuracy of the model via Leave-one-out-cross-validation and also find
the values of True Positive,True Negative,False Positive and False Negative
for the same metric.

PART 2
The matrix "argo_performance.csv" contains the accuracy values of 5 machine 
learning algorithms for 30 datasets. Use the "Friedman test" to check if the
hypothesis that these algorithms do not have large statistical differencies.
The validation should be done for a=[0.05, 0.1, 0.25].
'''
#-----------------------------------------------------------------------------#
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn import datasets,linear_model,metrics
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------#
#--------PART 1--------LEAVE ONE OUT CROSS VALIDATION-------------------------#
print('--------PART 1--------LEAVE ONE OUT CROSS VALIDATION------------------')
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
#X = breast_cancer.data[:, :2]  # we only take the first two features.
y = breast_cancer.target

x_mean = X.mean()
x_std = X.std()
X = (X - x_mean)/x_std

logreg = linear_model.LogisticRegression()
kfold = KFold(n_splits=len(X),random_state=7)

start_LOO = timer()
# generate LOO predictions
LOO_predictions = cross_val_predict(logreg, X, y, cv=X.shape[1])
end_LOO = timer()
LOO_time = (end_LOO - start_LOO)

LOO_score_start = timer()
cv_results = cross_val_score(logreg, X, y, cv=kfold)
LOO_score_end = timer()
LOO_score_time = (LOO_score_end - LOO_score_start)

cm=metrics.confusion_matrix(y, LOO_predictions)
tn, fp, fn, tp = cm.ravel()
#accuracy = (tp + tn) / (tp + fp + fn + tn)
#acc=metrics.accuracy_score(y, LOO_predictions)
print ('Model accuracy : '+ str(cv_results.mean()*100)+' %')
print ('True  negatives : '+ str(tn))
print ('True  positives : '+ str(tp))
print ('False negatives : '+ str(fn))
print ('False positives : '+ str(fp))
#-----------------------------------------------------------------------------#
#--------PART 2--------Friedman Test------------------------------------------#
print('--------PART 2--------Friedman Test-----------------------------------')
data=pd.read_csv('algo_performance.csv')
data1=data['C4.5'].tolist()
data2=data['1-NN'].tolist()
data3=data['NaiveBayes'].tolist()
data4=data['Kernel'].tolist()
data5=data['CN2'].tolist()
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
a=[0.05,0.1,0.25]
for i in range(0,len(a)):
    if p > a[i]:
    	print('Probably the same distribution, for a='+str(a[i]))
    else:
    	print('Probably different distributions, for a='+str(a[i]))
#-----------------------------------------------------------------------------#