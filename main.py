from collect_data import collect_data
from five_fold_cross_validation import *

#Create 5-core dataset
collect_data('dataset/')

metric = 'Pearson Correlation'
classification = 'ekman'
setting = 'ml'
print(metric)
predictive_algorithm(classification, metric, setting, False)
print('Results including emotions from reviews\' analysis')
predictive_algorithm(classification, metric, setting, True)

metric = 'Cosine Similarity'
classification = 'ekman'
print(metric)
predictive_algorithm(classification, metric, setting, False)
print('Results including emotions from reviews\' analysis')
predictive_algorithm(classification, metric, setting, True)