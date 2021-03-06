'''
Association Rules (Apriori algorithm)
'''
# =============================================================================
# Import libraries:
import pandas as pd                                     # For working with DataFrame objects
from mlxtend.preprocessing import TransactionEncoder    # For encoding dataset
from mlxtend.frequent_patterns import apriori           # The apriori algorithm
from mlxtend.frequent_patterns import association_rules # For producing association rules

# =============================================================================
# Load the dataset.
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# =============================================================================
# Use a TransactionEncoder object first to encode embedded Python lists into Numpy arrays.
# "We can transform this dataset into an array format suitable for typical machine learning APIs.
# Via the fit method, the TransactionEncoder learns the unique labels in the dataset, and via the transform method,
# it transforms the input dataset (a Python list of lists) into a one-hot encoded NumPy boolean array."
# Create TransactionEncoder object using TransactionEncoder() method.
te =TransactionEncoder()

# =============================================================================
# Fit and transform the TransactionEncoder object on the given dataset.
te_ary = te.fit(dataset).transform(dataset)

# =============================================================================
# Convert the encoded data into a pandas DataFrame.
# Make sure that column names are distinguished from the rest of the data,
# by passing column names (te.columns_) into the columns argument of the dataframe.
df = pd.DataFrame(te_ary, columns=te.columns_)

# =============================================================================
'''
mlxtend library has an efficient implementation of the apriori algorithm.
Apart from the dataset (as a DataFrame object) that is to be used in the  algorithm, the following parameters can be set:
    min_support  : Floating point value between 0 and 1 that indicates the minimum support required for an itemset to be selected.
    use_colnames : This allows to preserve column names for itemset making it more readable.
    max_len      : Max length of itemset generated. If not set, all possible lengths are evaluated.

For this project, no suggested values are provided for the parameters.
Note: The apriori() function returns a DataFrame object.
'''
# ADD COMMAND FOR APRIORI ALGORITHM HERE
apriori_results = apriori(df, min_support=0.6,use_colnames=True, max_len=None)

# =============================================================================
'''
Extract association rules from the apriori algorithm using the association_rules() function.
Frequent if-then associations called association rules can be extracted from the results of the apriori algorithm.
Apart from the resulting DataFrame object after running the apriori algorithm, the following parameters can be passed in the function:
    metric: can be set to 'confidence', 'lift', 'support', 'leverage' and 'conviction'.
    min_threshold: Minimal threshold for the evaluation metric, via the metric parameter, to decide whether a candidate rule is of interest.
    
Use 'support' for the metric parameter. Suggested value for min_threshold are not provided.
'''
# ADD COMMAND TO EXTRACT ASSOCIATION RULES FROM APRIORI ALGORITHM
frequent_itemsets = association_rules(apriori_results,metric='lift',min_threshold=1)

# =============================================================================
# ADD COMMAND HERE TO SORT FREQUENT ITEMSETS BY 'CONFIDENCE' AND PRINT THEM.
sorted_frequent_itemsets=frequent_itemsets.sort_values(by=['confidence'])
pd.set_option('display.expand_frame_repr', False)
print(sorted_frequent_itemsets)