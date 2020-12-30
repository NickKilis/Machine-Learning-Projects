'''
Bayesian learning (Naive Bayes algorithm)

From sklearn, we will import:
    'datasets'        : for loading data
    'model_selection' : package, which will help validate our results
    'metrics' package : for measuring scores
    'naive_bayes'     : package, for creating and using Naive Bayes classfier

We also need to import 'make_pipeline' from the 'pipeline' module.

We are working with text, so we need an appropriate package that shall vectorize words within our texts:
Import : 'TfidfVectorizer' from 'feature_extraction.text'.

'matplotlib.pyplot' and 'seaborn' are ncessary as well, for plotting the confusion matrix.
'''
# =============================================================================
from sklearn import datasets,metrics,model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn import pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Load text data.
textData = datasets.fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# =============================================================================
# Store features and target variable into 'X' and 'y'.
X = textData.data
y = textData.target

# =============================================================================
# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 0)

# =============================================================================
# We need to perform a transformation on the model that will later become 
# our Naive Bayes classifier. This transformation is text vectorization,
# using TfidfVectorizer().
# When you want to apply several transformations on a model, and an
# estimator at the end, you can use a 'pipeline'. This allows you to
# define a chain of transformations on your model, like a workflow.
# In this case, we have one transformer that we wish to apply (TfidfVectorizer)
# and an estimator afterwards (Multinomial Naive Bayes classifier).
textTokenizer = text.TfidfVectorizer(x_train)
clf = MultinomialNB()
alpha_values=[0.1,0.5,0.8,1,2,3]
alpha = alpha_values[0] # This is the smoothing parameter for Laplace/Lidstone smoothing
model = pipeline.Pipeline([('TextTokenizer', textTokenizer), ('NaiveBayes', clf)])
fit_prior_values = [True, False]
fit_prior=fit_prior_values[0]

model.set_params(NaiveBayes__alpha=alpha, NaiveBayes__fit_prior=fit_prior)

# Let's train our model.
model.fit(x_train, y_train)

# =============================================================================
# Ok, now let's predict output for the second subset
y_predicted = model.predict(x_test) 

# =============================================================================
# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'accuracy_score()', recall_score()', 'precision_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform 
# a type of averaging on the data. Use 'macro' for final results.
accuracy =  metrics.accuracy_score(y_test, y_predicted)
result = metrics.precision_recall_fscore_support(y_test, y_predicted, average='macro') 
precision = result[0]
recall = result[1]
f1 = result[2]
print("Accuracy: %f" % accuracy)
print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)

# =============================================================================
# In order to plot the 'confusion_matrix', first grab it from the 'metrics' module
# and then throw it within the 'heatmap' method from the 'seaborn' module.
confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
fitStr = ''
if fit_prior is True:
    fitStr = '_t'
else:
   fitStr = '_f'
plt.figure(figsize=(20,10))
ax = sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="OrRd", cbar=False,xticklabels=textData.target_names, yticklabels=textData.target_names)

plt.suptitle('Multinomial NB - Confusion matrix (a = %.1f) [Acc= %.3f,Prec = %.3f, Rec = %.3f, F1 = %.3f] with fit_prior= %s' % (alpha,accuracy, precision, recall, f1, str(fit_prior_values[0])), fontsize=18)
#plt.title('Multinomial NB - Confusion matrix (a = %.1f) [Prec = %.3f, Rec = %.3f, F1 = %.3f] with fit_prior= %s' % (alpha, precision, recall, f1, str(fit_prior_values[0])))
plt.xlabel('True output')
plt.ylabel('Predicted output')
plt.savefig('results/c_m_' + str(alpha) + fitStr + '.png')
plt.show()