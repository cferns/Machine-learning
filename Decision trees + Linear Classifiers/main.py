#Machine Learning - CS6350 - Homework 1
#Clinton Fernandes
#u1016390
#All solutions to Homework-1 question 3 and function calls are here

from __future__ import division
import treeFunctions
from collections import Counter
import timeit


all_start_time = timeit.default_timer()
A_start_time = timeit.default_timer()
print 'code is running'

#for A.1.b, A.1.d
training_filepath_A = './datasets/SettingA/training.data'
trainingMatrix, numExamples, numTrainingAttributes = treeFunctions.read_file(training_filepath_A)
depth_of_tree = 0
limiting_depth = 1000
to_be_visited = range(numTrainingAttributes)
last_col = treeFunctions.elements_of_column(trainingMatrix, len(trainingMatrix[0])-1)
list_of_labels = list(Counter(last_col))
majorityTraining_label = (Counter(last_col)).most_common(1)[0][0]
tree = treeFunctions.ID3_implementation(trainingMatrix, numTrainingAttributes, depth_of_tree, to_be_visited,
                                        majorityTraining_label, limiting_depth, method = 0)
classified_list_train = treeFunctions.classify_test(tree, trainingMatrix, list_of_labels, majorityTraining_label)
av_TrainingAccuracy = treeFunctions.test_accuracy(classified_list_train, trainingMatrix)
error_training = (100-av_TrainingAccuracy)
depth_list = treeFunctions.find_depths_of_tree(tree)
max_depth = max(depth_list)

#for A.1.c
test_filepath_A = './datasets/SettingA/test.data'
testMatrix, numTests, numTestAttributes = treeFunctions.read_file(test_filepath_A)
classified_list_test = treeFunctions.classify_test(tree, testMatrix, list_of_labels, majorityTraining_label)
av_TestAccuracy = treeFunctions.test_accuracy(classified_list_test, testMatrix)
error_testing = (100-av_TestAccuracy)

#for A.2.a

splitPath_A = './datasets/SettingA/CVSplits'
depth_range = [1,2,3,4,5,10,15,20]
a_CV_accuracy_list = []
a_CV_std_dev_list = []
for depth_limit in depth_range:
    av_CV_accuracy, std_dev = treeFunctions.cross_valid_Experiment(splitPath_A, depth_limit, method=0)
    a_CV_accuracy_list.append(av_CV_accuracy)
    a_CV_std_dev_list.append(std_dev)

#for A.2.b
best_accuracy = max(a_CV_accuracy_list)
max_index = a_CV_accuracy_list.index(best_accuracy)
best_depth = depth_range[max_index]

trainingMatrix, numExamples, numTrainingAttributes = treeFunctions.read_file(training_filepath_A)
depth_of_tree = 0
to_be_visited = range(numTrainingAttributes)
last_col = treeFunctions.elements_of_column(trainingMatrix, len(trainingMatrix[0])-1)
list_of_labels = list(Counter(last_col))
majorityTraining_label = (Counter(last_col)).most_common(1)[0][0]
limiting_depth = best_depth
method = 0
tree_A2b = treeFunctions.ID3_implementation(trainingMatrix, numTrainingAttributes, depth_of_tree, to_be_visited,
                                        majorityTraining_label, limiting_depth, method)
classified_list_test_A2b = treeFunctions.classify_test(tree_A2b, testMatrix, list_of_labels, majorityTraining_label)
a_accuracy_aft_CV = treeFunctions.test_accuracy(classified_list_test_A2b, testMatrix)

print '\n --------------- Setting A solutions: --------------- \n'
print 'A.1.a : Answer in Report'
print 'A.1.b : Error of decision tree on the SettingA/training.data file: %i' %error_training,'%'
print 'A.1.c : Error of decision tree on the SettingA/test.data file: %i' %error_testing,'%'
print 'A.1.d : Maximum depth of decision tree: %i' %max_depth
print 'A.2.a : Average cross-validation accuracy and standard deviation for each depth: '
print 'depth       | accuracy |            std_deviation'
for row in zip(depth_range, a_CV_accuracy_list, a_CV_std_dev_list):
    print '        '.join(str(e) for e in row)
print 'A.2.b : Accuracy of decision tree on the SettingA/test.data file: (with best depth = %i) is: %f' %(best_depth,a_accuracy_aft_CV)

A_stop_time = timeit.default_timer()
print 'Time taken for Setting A code to run: ', A_stop_time - A_start_time

#######################################################################################################################

#######################################################################################################################
B_start_time = timeit.default_timer()

#for B.1.a
training_filepath_B = './datasets/SettingB/training.data'
trainingBMatrix, numExamples, numTrainingAttributes = treeFunctions.read_file(training_filepath_B)
depth_of_tree = 0
limiting_depth = 1000
to_be_visited = range(numTrainingAttributes)
last_col = treeFunctions.elements_of_column(trainingBMatrix, len(trainingBMatrix[0])-1)
list_of_labels = list(Counter(last_col))
majorityTrainingB_label = (Counter(last_col)).most_common(1)[0][0]
tree = treeFunctions.ID3_implementation(trainingBMatrix, numTrainingAttributes, depth_of_tree, to_be_visited,
                                        majorityTrainingB_label, limiting_depth, method = 0)

classified_list_train = treeFunctions.classify_test(tree, trainingBMatrix, list_of_labels, majorityTrainingB_label)
av_TrainingAccuracy = treeFunctions.test_accuracy(classified_list_train, trainingBMatrix)
error_trainingB = (100-av_TrainingAccuracy)
depth_list = treeFunctions.find_depths_of_tree(tree)
max_depth = max(depth_list)
# for B.1.b
test_filepath_B = './datasets/SettingB/test.data'
testMatrix, numTests, numTestAttributes = treeFunctions.read_file(test_filepath_B)
classified_list_test = treeFunctions.classify_test(tree, testMatrix, list_of_labels, majorityTrainingB_label)
av_TestAccuracy = treeFunctions.test_accuracy(classified_list_test, testMatrix)
error_testingB = (100-av_TestAccuracy)
# for B.1.c
training_filepath_A = './datasets/SettingA/training.data'
trainAMatrix, numTests, numTestAttributes = treeFunctions.read_file(training_filepath_A)
classified_list_test = treeFunctions.classify_test(tree, trainAMatrix, list_of_labels, majorityTrainingB_label)
av_TestAccuracy = treeFunctions.test_accuracy(classified_list_test, trainAMatrix)
error_trainingA = (100-av_TestAccuracy)
# for B.1.d
test_filepath_A = './datasets/SettingA/test.data'
testBMatrix, numTests, numTestAttributes = treeFunctions.read_file(test_filepath_A)
classified_list_test = treeFunctions.classify_test(tree, testBMatrix, list_of_labels, majorityTrainingB_label)
av_TestAccuracy = treeFunctions.test_accuracy(classified_list_test, testBMatrix)
error_testingA = (100-av_TestAccuracy)

#for B.2.a
splitPath_B = './datasets/SettingB/CVSplits'
depth_range = [1,2,3,4,5,10,15,20]
b_CV_accuracy_list = []
b_CV_std_dev_list = []

for depth_limit in depth_range:
    av_CV_accuracy, std_dev = treeFunctions.cross_valid_Experiment(splitPath_B, depth_limit, method=0)
    b_CV_accuracy_list.append(av_CV_accuracy)
    b_CV_std_dev_list.append(std_dev)

#for B.2.b
best_accuracy = max(b_CV_accuracy_list)
max_index = b_CV_accuracy_list.index(best_accuracy)
best_depth = depth_range[max_index]

trainingBMatrix, numExamples, numTrainingAttributes = treeFunctions.read_file(training_filepath_B)
depth_of_tree = 0
to_be_visited = range(numTrainingAttributes)
last_col = treeFunctions.elements_of_column(trainingBMatrix, len(trainingBMatrix[0])-1)
list_of_labels = list(Counter(last_col))
majorityTraining_label = (Counter(last_col)).most_common(1)[0][0]
limiting_depth = 4
method = 0
tree_B2b = treeFunctions.ID3_implementation(trainingBMatrix, numTrainingAttributes, depth_of_tree, to_be_visited,
                                        majorityTraining_label, limiting_depth, method)
classified_list_test_B2b = treeFunctions.classify_test(tree_B2b, testMatrix, list_of_labels, majorityTraining_label)
b_accuracy_aft_CV = treeFunctions.test_accuracy(classified_list_test_B2b, testMatrix)

print '\n --------------- Setting B solutions: --------------- \n'
print 'B.1.a : Error of decision tree on the SettingB/training.data : %.3f' %error_trainingB,'%'
print 'B.1.b : Error of decision tree on the SettingB/test.data file : %.3f'%error_testingB,'%'
print 'B.1.c : Error of decision tree on the SettingA/training.data file : %.3f ' %error_trainingA,'%'
print 'B.1.d : Error of decision tree on the SettingA/test.data file : %.3f' %error_testingA,'%'
print 'B.1.e : Maximum depth of decision tree : %i' %max_depth
print 'B.2.a : Cross-validation accuracy and standard deviation for each depth is below. Best depth : %i' %best_depth
print 'depth       | accuracy |            std_deviation'
for row in zip(depth_range, b_CV_accuracy_list, b_CV_std_dev_list):
    print '        '.join(str(e) for e in row)
print 'B.2.b : Accuracy of decision tree on the SettingB/test.data file : %.3f ' %b_accuracy_aft_CV,'%'
B_stop_time = timeit.default_timer()
print 'Time taken for Setting B code to run: ', B_stop_time - B_start_time

#######################################################################################################################

#######################################################################################################################
C_start_time = timeit.default_timer()
#for C.2
splitPath_C = './datasets/SettingC/CVSplits'
depth_range = [1,2,3,4,5,10,15,20]
c1_CV_accuracy_list = []
c1_CV_std_dev_list = []
c2_CV_accuracy_list = []
c2_CV_std_dev_list = []
c3_CV_accuracy_list = []
c3_CV_std_dev_list = []

for depth_limit in depth_range:
    av_CV_accuracy, std_dev = treeFunctions.cross_valid_Experiment(splitPath_C, depth_limit, method = 1)
    c1_CV_accuracy_list.append(av_CV_accuracy)
    c1_CV_std_dev_list.append(std_dev)

for depth_limit in depth_range:
    av_CV_accuracy, std_dev = treeFunctions.cross_valid_Experiment(splitPath_C, depth_limit, method = 2)
    c2_CV_accuracy_list.append(av_CV_accuracy)
    c2_CV_std_dev_list.append(std_dev)

for depth_limit in depth_range:
    av_CV_accuracy, std_dev = treeFunctions.cross_valid_Experiment(splitPath_C, depth_limit, method = 3)
    c3_CV_accuracy_list.append(av_CV_accuracy)
    c3_CV_std_dev_list.append(std_dev)

#for C.3
training_filepath_C = './datasets/SettingC/training.data'
trainingMatrix, numExamples, numTrainingAttributes = treeFunctions.read_file(training_filepath_C)
depth_of_tree = 0
limiting_depth = 1000
to_be_visited = range(numTrainingAttributes)
last_col = treeFunctions.elements_of_column(trainingMatrix, len(trainingMatrix[0])-1)
list_of_labels = list(Counter(last_col))
majorityTraining_label = (Counter(last_col)).most_common(1)[0][0]
tree = treeFunctions.ID3_implementation(trainingMatrix, numTrainingAttributes, depth_of_tree, to_be_visited,
                                        majorityTraining_label, limiting_depth, method = 3)
test_filepath_C = './datasets/SettingC/test.data'
testCMatrix, numTests, numTestAttributes = treeFunctions.read_file(test_filepath_C)
classified_list_test = treeFunctions.classify_test(tree, testCMatrix, list_of_labels, majorityTraining_label)
av_CTestAccuracy = treeFunctions.test_accuracy(classified_list_test, testCMatrix)

print '\n --------------- Setting C solutions: --------------- \n'
print 'C.1 : Answer in report : '
print 'C.2 : Accuracy for each method and the standard deviation : '
print 'depth     | m1-accuracy     | m1-std_deviation     | m2-accuracy     | m2-std_deviation     | m3-accuracy     | m3-std_deviation'
for row in zip(depth_range, c1_CV_accuracy_list, c1_CV_std_dev_list, c2_CV_accuracy_list,c2_CV_std_dev_list, c3_CV_accuracy_list,c3_CV_std_dev_list):
    print '        '.join(str(e) for e in row)
print 'C.3 : Using the best method, Accuracy of tree on SettingC/test.data : %.3f' % av_CTestAccuracy, '%'
C_stop_time = timeit.default_timer()
print 'Time taken for Setting C code to run: ', C_stop_time - C_start_time

all_stop_time = timeit.default_timer()
print '\nTime taken for whole code to run: ', all_stop_time - all_start_time
#######################################################################################################################
