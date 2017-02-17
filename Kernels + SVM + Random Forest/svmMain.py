import svmFunctions

handwriting_train_MAT = svmFunctions.read_file('./data/handwriting/train.data')
handwriting_train_LABELS = svmFunctions.read_file('./data/handwriting/train.labels')
handwriting_test_MAT = svmFunctions.read_file('./data/handwriting/test.data')
handwriting_test_LABELS = svmFunctions.read_file('./data/handwriting/test.labels')
#
madelon_train_MAT = svmFunctions.read_file('./data/madelon/madelon_train.data')
madelon_train_LABELS = svmFunctions.read_file('./data/madelon/madelon_train.labels')
madelon_test_MAT = svmFunctions.read_file('./data/madelon/madelon_test.data')
madelon_test_LABELS = svmFunctions.read_file('./data/madelon/madelon_test.labels')

print '3.1. Support Vector Machines:'
print '\n3.1.1: solution'
epochs_31 = 7; C_31 = 1;gamma_31 = 0.01
w_31 = svmFunctions.SVM(handwriting_train_MAT, handwriting_train_LABELS, epochs_31, C_31, gamma_31)
pred_train_list_31 = svmFunctions.prediction(handwriting_train_MAT, handwriting_train_LABELS, w_31)
pred_test_list_31 = svmFunctions.prediction(handwriting_test_MAT, handwriting_test_LABELS, w_31)
acc_train = svmFunctions.accuracyMETRIC(handwriting_train_LABELS, pred_train_list_31)
acc_test = svmFunctions.accuracyMETRIC(handwriting_test_LABELS, pred_test_list_31)
print '\tTraining accuracy:', acc_train,'%'
print '\tTest accuracy', acc_test,'%'

print '\n3.1.2: solution'
Clist = [2,2**-3,2**-4,2**-5,2**-6,2**-7]; Rlist = [0.1,0.001,0.0001]; num_folds = 5; epochs_32 = 1
store,bestGamma,bestC,bestAccuracy = svmFunctions.crossValidation(madelon_train_MAT,madelon_train_LABELS,Clist,Rlist,num_folds,epochs_32)
epochs = 1
w_32 = svmFunctions.SVM(madelon_train_MAT, madelon_train_LABELS, epochs, bestC, bestGamma)
pred_train_list_32 = svmFunctions.prediction(madelon_train_MAT, madelon_train_LABELS, w_32)
pred_test_list_32 = svmFunctions.prediction(madelon_test_MAT, madelon_test_LABELS, w_32)
acc_train = svmFunctions.accuracyMETRIC(madelon_train_LABELS, pred_train_list_32)
acc_test = svmFunctions.accuracyMETRIC(madelon_test_LABELS, pred_test_list_32)
print '\tRange of gamma(0): ', Rlist
print '\tRange of C: ',Clist
print '\tBest gamma(0) : ', bestGamma
print '\tBest C: ', bestC
print '\tTraining accuracy:', acc_train,'%'
print '\tTest accuracy', acc_test,'%'

print '\n3.1.3'
precision_31_train,recall_31_train , f1_31_train = svmFunctions.f1METRIC(handwriting_train_LABELS, pred_train_list_31)
precision_31_test,recall_31_test, f1_31_test = svmFunctions.f1METRIC(handwriting_test_LABELS, pred_test_list_31)
precision_32_train,recall_32_train, f1_32_train = svmFunctions.f1METRIC(madelon_train_LABELS, pred_train_list_32)
precision_32_test,recall_32_test, f1_32_test = svmFunctions.f1METRIC(madelon_test_LABELS, pred_test_list_32)
print '\tOn the Handwritten data (Q 3.1.1):'
print '\t\tOn training:'
print '\t\t\tPrecision: ',precision_31_train
print '\t\t\tRecall: ',recall_31_train
print '\t\t\tF1 score: ',f1_31_train
print '\t\tOn test:'
print '\t\t\tPrecision: ',precision_31_test
print '\t\t\tRecall: ',recall_31_test
print '\t\t\tF1 score: ',f1_31_test
#
print '\tOn the Madelon data (Q 3.1.2):'
print '\t\tOn training:'
print '\t\t\tPrecision: ',precision_32_train
print '\t\t\tRecall: ',recall_32_train
print '\t\t\tF1 score: ',f1_32_train
print '\t\tOn test:'
print '\t\t\tPrecision: ',precision_32_test
print '\t\t\tRecall: ',recall_32_test
print '\t\t\tF1 score: ',f1_32_test