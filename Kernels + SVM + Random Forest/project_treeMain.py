import percepFunctions
import treeFunctions
import projectFunctions
import percepFunctions
from collections import Counter
import timeit
import crossValidationExperiments
import numpy as np

startTime = timeit.default_timer()

train_filepath = './data/data-splits/data.train'
#train_filepath = './data/data_temp/data.train'
train_matrix, train_actual_labels = treeFunctions.readFile_multi(train_filepath,'w')
test_filepath = './data/data-splits/data.test'
#test_filepath = './data/data_temp/data2.train'
test_matrix, test_actual_labels = treeFunctions.readFile_multi(test_filepath,'w')
eval_anon_filepath = './data/data-splits/data.eval.anon'
#eval_anon_filepath = './data/data_temp/data3.train'
eval_matrix, eval_actual_labels = treeFunctions.readFile_multi(eval_anon_filepath,'w')

train_matrix, test_matrix, eval_matrix  = projectFunctions.featureTransformation(train_matrix, test_matrix, eval_matrix,1)

print '\n------------------ Decision tree: ------------------'
num_TrainingFEATURES = len(train_matrix[0])
depth_of_tree = 0; limiting_depth = 100;
to_be_visited = range(num_TrainingFEATURES)
major_trainLABEL = (Counter(train_actual_labels)).most_common(1)[0][0]
train_MATnLBLS = projectFunctions.features_n_labels_matrix(train_matrix, train_actual_labels)
tree = treeFunctions.ID3_implementation(train_MATnLBLS, num_TrainingFEATURES, depth_of_tree, to_be_visited,
                                        major_trainLABEL, limiting_depth, method = 0); print tree
depth_list = treeFunctions.find_depths_of_tree(tree); max_depth = max(depth_list)

list_of_labels = list(Counter(train_actual_labels))
predicted_train_list = treeFunctions.predictFUNCTION(tree, train_matrix, list_of_labels, major_trainLABEL)
predicted_test_list = treeFunctions.predictFUNCTION(tree, test_matrix, list_of_labels, major_trainLABEL)
predicted_eval_list = treeFunctions.predictFUNCTION(tree, eval_matrix, list_of_labels, major_trainLABEL)
#
train_accuracy = projectFunctions.accuracyMETRIC(train_actual_labels, predicted_train_list)
train_f1_score = projectFunctions.f1METRIC(train_actual_labels, predicted_train_list)
test_accuracy = projectFunctions.accuracyMETRIC(test_actual_labels, predicted_test_list)
test_f1_score = projectFunctions.f1METRIC(test_actual_labels, predicted_test_list)
eval_accuracy = projectFunctions.accuracyMETRIC(eval_actual_labels, predicted_eval_list)
eval_f1_score = projectFunctions.f1METRIC(eval_actual_labels, predicted_eval_list)

solution_filename = raw_input('Enter x__x in ./solutions_log/x__x_solutions.csv: ')
projectFunctions.write_solutions('decision tree',predicted_eval_list,'./solutions_log/solutions/'+solution_filename+'.solutions.csv')
#np.save('trees.npy', tree)
#read_tree = np.load('trees.npy').item()
#
print 'Train Accuracy: '
print train_accuracy
print 'Train f1: '
print train_f1_score
print 'Test Accuracy: '
print test_accuracy
print 'Test f1: '
print test_f1_score
print 'Eval Accuracy: '
print eval_accuracy
print 'Eval f1: '
print eval_f1_score

stopTime = timeit.default_timer()
print 'time: ', stopTime - startTime