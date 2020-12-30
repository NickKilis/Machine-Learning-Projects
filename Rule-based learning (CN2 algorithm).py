'''
RULE-BASED LEARNING (CN2 ALGORITHM)

For this project, the only thing that we will need to import is the "Orange" library.
However, before importing it, you must first install the library into Python.
Read the instructions on how to do that (it might be a bit trickier than usual!)

'''
# IMPORT LIBRARY HERE (trivial but necessary...)
import Orange
# =============================================================================
#from Orange.data import Table
#from Orange.classification import CN2Learner, CN2UnorderedLearner

# Load 'wine' dataset
#wineData = datasets.load_wine()
wineData = Orange.data.Table('wine')
# =============================================================================

# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
learner_unordered= Orange.classification.rules.CN2UnorderedLearner()
learner_ordered1 = Orange.classification.rules.CN2Learner()
learner_ordered2 = Orange.classification.rules.CN2Learner()
#learner_ordered_laplace = Orange.classification.CN2OrderedLearner(evaluator=Evaluator_Laplace())

# =============================================================================
# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up), 
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
case = 1
bw=[5]
mce=[7,15]
mrl=[2,5]
if case==1:
    entropy = Orange.classification.rules.EntropyEvaluator
    learner_ordered1.rule_finder.search_algorithm.evaluator=[entropy, "Entropy"]
    learner_ordered1.rule_finder.search_algorithm.beam_width = bw[0]
    learner_ordered1.rule_finder.general_validator.min_covered_examples = mce[0]
    learner_ordered1.rule_finder.general_validator.max_rule_length = mrl[0]
elif case==2:
    laplace = Orange.classification.rules.LaplaceAccuracyEvaluator
    learner_unordered.rule_finder.search_algorithm.evaluator=[laplace, "Laplace"]
    learner_unordered.rule_finder.search_algorithm.beam_width = bw[0]
    learner_unordered.rule_finder.general_validator.min_covered_examples =  mce[1]
    learner_unordered.rule_finder.general_validator.max_rule_length = mrl[1]
elif case==3:
    laplace = Orange.classification.rules.LaplaceAccuracyEvaluator
    learner_ordered2.rule_finder.search_algorithm.evaluator=[laplace, "Laplace"]
    learner_ordered2.rule_finder.search_algorithm.beam_width = bw[0]
    learner_ordered2.rule_finder.general_validator.min_covered_examples = mce[0]
    learner_ordered2.rule_finder.general_validator.max_rule_length = mrl[0]

# =============================================================================
# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets, 
# then train the model, and produce results.
# So, simply call the CrossValidation() function from the 'testing' library
# and use as input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.
#results = Orange.evaluation.testing.CrossValidation(wineData,[learner_unordered])
if case==1:
    results = Orange.evaluation.CrossValidation(wineData, [learner_ordered1],k=5)
if case==2:
    results = Orange.evaluation.CrossValidation(wineData, [learner_unordered],k=5)
if case==3:
    results = Orange.evaluation.CrossValidation(wineData, [learner_ordered2],k=5)

# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform 
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER 
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
print('# ============================================================================ #') 
print('Results:')
print("Accuracy : %.3f" % Orange.evaluation.CA(results)[0])
print("Precision: %.3f" % Orange.evaluation.Precision(results,average='macro')[0])
print("Recall   : %.3f" % Orange.evaluation.Recall(results,average='macro')[0])
print("F1       : %.3f" % Orange.evaluation.F1(results,average='macro')[0])

# =============================================================================
# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================
print('# ============================================================================ #')
if case==1:      
    print('Learner type:                     ' + 'ordered')
    print('Evaluator:                        ' + 'entropy')
    print('Beam width:                       ' + str(bw[0]))
    print('Minimum examples covered:         ' + str(mce[0]))
    print('Maximum rule length:              ' + str(mrl[0]))
    print('Average:                          ' + 'macro')
    print('# ============================================================================ #')
    print('Rules:')
if case==2:      
    print('Learner type:                     ' + 'unordered')
    print('Evaluator:                        ' + 'laplace')
    print('Beam width:                       ' + str(bw[0]))
    print('Minimum examples covered:         ' + str(mce[1]))
    print('Maximum rule length:              ' + str(mrl[1]))
    print('Average:                          ' + 'macro')
    print('# ============================================================================ #')
    print('Rules:')
if case==3:      
    print('Learner type:                     ' + 'ordered')
    print('Evaluator:                        ' + 'laplace')
    print('Beam width:                       ' + str(bw[0]))
    print('Minimum examples covered:         ' + str(mce[0]))
    print('Maximum rule length:              ' + str(mrl[0]))
    print('Average:                          ' + 'macro')
    print('# ============================================================================ #')
    print('Rules:')

# ADD COMMAND TO TRAIN THE LEARNER HERE
if case==1:
    classifier = learner_ordered1(wineData)
if case==2:
    classifier = learner_unordered(wineData)
if case==3:
    classifier = learner_ordered2(wineData)
# =============================================================================

# Now we can print the derived rules. To do that, we need to iterate through 
# the 'rules_list' of our classifier.
for rule in classifier.rule_list:
    print(rule)
print('# ============================================================================ #')
    