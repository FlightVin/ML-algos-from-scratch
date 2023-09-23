#!/bin/bash

# reading the test file path
test_file_path="$1"

# importing the required python function and running it
python3 -c "
from run_knn_script import *

train_dataset = np.load('train.npy', allow_pickle=True)
test_dataset = np.load('$test_file_path', allow_pickle=True)

myKNN = KNN(k = 10, encoding = 2, weigh = True, dist_metric = 'euclidean', 
           validation_setting = False,
           train_data = train_dataset, test_data = test_dataset)
accuracy, precision, recall, f1 = myKNN.run()

print(accuracy, f1, recall, precision)

print(f'\nMetrics:')
print(f'\tAccuracy: {accuracy:.8f}')
print(f'\tF1 Score: {f1:.8f}')
print(f'\tRecall: {recall:.8f}')
print(f'\tPrecision: {precision:.8f}')
"