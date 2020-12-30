#-----------------------------------------------------------------------------------------------------------#
#                                                                                                           #
# DATASET : "hate_tweets.csv"                                                                               #
# The style in a tweet for this dataset can lie in 3 categories:                                            #
#                                                               1."hate_speech"         label=0             #
#                                                               2."offensive language"  label=1             #
#                                                               3."neither"             label=2             #
# 1.Select a ML algorithm that can distinguish the style of a tweet, as far as the aggression of the writer.#
# 2.Run several experiments that show the combination of parameters used in the algorithm.                  #
# 3.Calculate the metrics : "Accuracy", "Precision", "Recall", "F1" of the model you created.               #
# 4.Write a report, containing the description and the results.                                             #
#                                                                                                           #
# HOW TO USE:                                                                                               #
# 1.Select - clean_data=True  ,in order to clean the text by removing useless information.                  #
#          In this case the data will be cleaned (from function : clean_string) and                         # 
#          stored in a new csv file(from function : write_csv) named clean_hate_tweets.csv                  #
# 2.Select - clean_data=False ,if you already cleaned the data.                                             #
#          In this case the cleaned data will be separated from the classes,splitted into a train(~67%) and # 
#          test(~33%) dataset stratified by the variable class and go through the pipeline. The pipeline    #
#          contains a textTokenizer and the algorithm MultinomialNB in that order. The pipeline produces    #
#          10 cases, which are the combination of 2 variables "fit_prior" and "alpha". For each case we     #
#          are interested how the model scores for the metrics "accuracy","precision","recall" and "F1".    #
#          The program also plots the confusion matrix for visualizing the mistakes between the 3 classes.  #
#          The results are stored in an xls file called "results" that contains 2 excel sheets.             #
#                                                                                                           #
#-----------------------------------------------------------------------------------------------------------#
import re
import time
import xlsxwriter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import pipeline
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")
#============================================================================================================
# START THE OVERALL TIMER
start_program = time.perf_counter()
#============================================================================================================
xls_writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
#============================================================================================================
# Select first "clean_data=True" and then "clean_data=False"
def main(clean_data=True):
    # Read the data from the csv file
    if clean_data is True:
        # Read original data, that need to be cleaned
        data = pd.read_csv("hate_tweets.csv")
        col_start = 1
    else:
        # Read cleaned data, that don't need further cleaning
        data = pd.read_csv("clean_hate_tweets.csv")
        col_start = 5
    # Get number for rows and columns
    rows = data.shape[0]
    cols = data.shape[1]
    # Separate the class from the data
    if clean_data is True:
        Y = data.iloc[0:rows , col_start:cols-1].values
        X = data.iloc[0:rows, cols-1].values
        # Clean and write them to new csv
        write_csv(X, Y)
    else:
        Y = data.iloc[0:rows , col_start:cols-1].values
        Y = Y.flatten().tolist()
        X = data.iloc[0:rows, cols-1].values.tolist()
        data_info = pd.DataFrame(columns=['Total Samples', 'Hate Speech', 'Offensive Language', 'Neither'])
        hs = 0
        ol = 0
        ne = 0
        for i in range(0, len(X)):
            if Y[i] == 0:
                hs += 1
            elif Y[i] == 1:
                ol += 1
            else:
                ne += 1
        data_info = data_info.append({ 'Total Samples': rows, 'Hate Speech': hs, 'Offensive Language': ol, 'Neither': ne}, ignore_index=True)
        data_info.to_excel(xls_writer, sheet_name='Dataset info')
        # Split data into train and test datasets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state = 0)
        x_train_length=len(x_train)
        x_test_length=len(x_test)
        test_size_percentage=100*x_test_length/x_train_length
        print('You have selected tha dataset Hate tweets\ndivided it into a trainset('+str(100-test_size_percentage)+'%) \nand a testset('+str(test_size_percentage)+'%).')
        # Train and predict using Multinomial Naive Bayes algorithm
        NB(x_train, y_train, x_test, y_test)
#============================================================================================================
# Create a function for calculating the Confusion Matrix and showing the plot
def computeAndPlotCM(y_test,y_predicted,fit_prior,alpha, accuracy, precision, recall, f1):
    cm=confusion_matrix(y_test,y_predicted)
    class_names=['hate','offensive','neither']
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names )
    fig = plt.figure(figsize = (8,8))
    fitStr = ''
    if fit_prior is True:
        fitStr = '_t'
    else:
        fitStr = '_f'
    plt.title('Multinomial NB - Confusion matrix (a = %.1f) [Acc= %.2f, Prec = %.2f, Rec = %.2f, F1 = %.2f] with fit_prior= %s' % (alpha, accuracy, precision, recall, f1, str(fit_prior)))      
    Ηeatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.xlabel('True output')
    plt.ylabel('Predicted output')
    plt.show()
    return cm
# Naive Bayes
def NB(x_train, y_train, x_test, y_test):
    # Create pipeline
    textTokenizer = text.TfidfVectorizer()
    clf = MultinomialNB()
    pipeline_model = pipeline.Pipeline([('TextTokenizer', textTokenizer), ('NaiveBayes', clf)])
    fit_prior_values = [True, False]
    # Initialize results dataframe
    results = pd.DataFrame(columns=['Uniform Probabilites', 'Alpha', 'Accuracy', 'Precision', 'Recall', 'F1'])
    # Naive Bayes
    for alpha in range(1, 10, 2):
        # Set alpha
        for fit_prior in fit_prior_values:
            # Set parameters
            pipeline_model.set_params(NaiveBayes__alpha=alpha, NaiveBayes__fit_prior=fit_prior)
            # Train model
            pipeline_model.fit(x_train, y_train)
            # Predict on test data
            y_predicted = pipeline_model.predict(x_test)
            # Compute metrics for precision, recall, fscore
            result = metrics.precision_recall_fscore_support(y_test, y_predicted, average='macro', labels=np.unique(y_predicted))
            accuracy = metrics.accuracy_score(y_test, y_predicted)
            # Set metric scores
            precision = result[0]
            recall = result[1]
            f1 = result[2]
            # If prior is false, a uniform prior is used
            prior = "True"
            if fit_prior is True:
                prior = "False"
            results = results.append({ 'Uniform Probabilites': prior, 'Alpha': alpha, 'Accuracy': float("%0.3f"%accuracy),
                          'Precision': float("%0.3f"%precision), 'Recall':  float("%0.3f"%recall), 'F1': float("%0.3f"%f1)}, ignore_index=True)
            # Confusion matrix
            confusionMatrix=computeAndPlotCM(y_test, y_predicted,fit_prior,alpha, accuracy, precision, recall, f1)
    print('The results organised are the following:\n',results)
    # Store the results in an xls file
    results.to_excel(xls_writer, sheet_name='Multinomial Naive Bayes')
    # Write all the results
    xls_writer.save()
#============================================================================================================
# Remove substring and special characters
def clean_string(string):
    # Remove RT tags
    pattern = '%s(.*?)%s' % (re.escape("RT"), re.escape(":"))
    s = re.sub(pattern, "", string)
    pattern = '%s(.*?)%s' % (re.escape("@"), re.escape(" "))
    s = re.sub(pattern, "", s)
    # Remove &amp; substrings
    s = re.sub(r"&amp;", "", s)
    # Remove emoticons
    pattern = '%s(.*?)%s' % (re.escape("&#"), re.escape(";"))
    s = re.sub(pattern, " ", s)
    # Remove links
    s = re.sub(r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", " ", s)
    # Remove special characters
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s)
    # Remove duplicate spaces
    s = re.sub(' +', ' ', s)
    # Remove trailing and duplicate spaces
    s = s.strip()
    # Check for empty strings
    if len(s) == 0:
        return "", False
    return s, True
#============================================================================================================
# Clean X and write new data to csv
def write_csv(X, Y):
    valid = []
    for i in range(0, len(X)):
        X[i], valid_string = clean_string(X[i])
        # Add valid id to list of ids
        if valid_string is True:
            valid.append(i)
    results = pd.DataFrame(columns=['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet'])
    for index in range(0, len(valid)):
        i = valid[index]
        results = results.append({ 'count': Y[i][0], 'hate_speech': Y[i][1],
                          'offensive_language': Y[i][2],
                          'neither': Y[i][3], 'class':  Y[i][4],
                          'tweet': X[i]}, ignore_index=True)
    # Write all the results
    results.to_csv('clean_hate_tweets.csv', sep=',', index=True)
#============================================================================================================
if __name__ == "__main__":
    main()
#============================================================================================================
# FIND HOW MUCH TIME HAS ELAPSED
end_program = time.perf_counter()
elapsed_program = end_program - start_program
# The program ends in 4.359966899999904 seconds having the cleaned data(clean_data=True) and
# in 82.05285270000013 seconds without having the cleaned data(clean_data=False).
print('The program finished in : '+ str(elapsed_program)+' seconds.') 
#============================================================================================================