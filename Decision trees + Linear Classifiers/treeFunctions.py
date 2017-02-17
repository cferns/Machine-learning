#Machine Learning - CS6350 - Homework 1
#Clinton Fernandes
#u1016390
# All the defined functions are here

from __future__ import division
from collections import Counter
import math
import csv
import copy
import timeit
import os
import numpy as np
from math import sqrt

def read_file(filepath):
    file = open(filepath, 'rb')
    data = csv.reader(file, delimiter=',')
    matrix = [row for row in data]
    num_examples = len(matrix)
    num_attributes = len(matrix[0]) - 1
    return matrix, num_examples, num_attributes

def trim_dataset(orig_data, colindex, valuename):
    new_data = []
    for i in range(len(orig_data)):
        if orig_data[i][colindex] == valuename:
            temp___row = orig_data[i]
            del temp___row[colindex]
            new_data.append(temp___row)
    return new_data

def elements_of_column(big_sdata, col_index):
    element_list = []
    for i in range(len(big_sdata)):
        element_list.append(big_sdata[i][col_index])
    return element_list

def last_output_elements(string_name,attr_index, matrix):
    list_last_els = []
    if string_name == 'all':
        last_col_index = len(matrix[1])-1
        for i in range(len(matrix)):
            list_last_els.append(matrix[i][last_col_index])
    else:
        last_col_index = len(matrix[1]) - 1
        for i in range(len(matrix)):
            if matrix[i][attr_index] == string_name:
                list_last_els.append(matrix[i][last_col_index])
    return list_last_els

def current_entropy(list_):
    pi_list,dummy,dummy = list_of_pi_s(list_)
    curr_ent = entropy(pi_list)
    return curr_ent

def entropy(list_of_pi):
    sum_e = 0
    for pi in list_of_pi:
        sum_e = sum_e - pi*math.log(pi,2)
    return sum_e

def list_of_pi_s(list_):
    v = Counter(list_)
    list_var = []
    list_var_counts = []
    list_probabilities = []
    sum = len(list_)
    for item in v:
        list_var.append(item)
        list_var_counts.append(v[item])
        list_probabilities.append(v[item]/sum)
    return list_probabilities, list_var, list_var_counts

def expected_entropy(big_sdata,num_attributes):
    exp_entropy_of_all_attrb = []
    for j in range(num_attributes):
        column_list = [item[j] for item in big_sdata]
        pi_list, list_var, list_var_counts = list_of_pi_s(column_list)
        curr_expected_entropy = 0
        for vcount in range(len(list_var)):
            prob_occur = pi_list[vcount]
            list_last_els = last_output_elements(list_var[vcount], j, big_sdata)
            list_probs, dummy, dummy = list_of_pi_s(list_last_els)
            entropy_ = entropy(list_probs)
            curr_expected_entropy+=  entropy_*prob_occur
        exp_entropy_of_all_attrb.append(curr_expected_entropy)
    return exp_entropy_of_all_attrb

def best_information_gain(curr_etrp, exp_etrp_list):
    inf_gain_list = []
    for i in range(len(exp_etrp_list)):
        igain = 0
        igain = curr_etrp - exp_etrp_list[i]
        inf_gain_list.append(igain)
    best_igain = max(inf_gain_list)
    best_igain_index = inf_gain_list.index(best_igain)
    return best_igain, best_igain_index

def ID3_all_examples(big_sdata, num_attributes):
    last_el_list = last_output_elements('all', 0, big_sdata)
    curr_entropy = current_entropy(last_el_list)
    exp_entropy_of_all_attrb = expected_entropy(big_sdata,num_attributes)
    best_info_gain, best_igain_index = best_information_gain(curr_entropy,exp_entropy_of_all_attrb)
    return best_info_gain, best_igain_index

def find_depths_of_tree(tree_dict, depth = 0, depth_list=[]):
    temp_depth = copy.deepcopy(depth)
    for each in tree_dict.keys():
        if type(tree_dict[each]) is dict:
            temp_dict = tree_dict[each]
            if isinstance(each, int):
                temp_depth += 1
            find_depths_of_tree(temp_dict, temp_depth)
        else:
            depth_list.append(temp_depth)
    return depth_list

def classify_test(dict, testMatrix, list_of_labels, majority_label):
    classified = []
    num_tests = len(testMatrix)
    for i in range(num_tests):
        test_ex_line = testMatrix[i]
        tempDict = copy.deepcopy(dict)
        a = 1
        while a == 1 :
            if tempDict in list_of_labels:
                break
            tree_root_index = tempDict.keys()[0]

            tempDict = tempDict.get(tree_root_index)
            val_in_test_at_root_index = test_ex_line[tree_root_index]
            '''
            keys_ = list(tempDict.keys())
            if val_in_test_at_root_index not in keys_:
                tempDict = majority_label
                break
            '''
            tdc = tempDict
            tempDict =  tempDict.get(val_in_test_at_root_index)
            if tempDict == None:
                #k = list(tdc.keys())
                #f = list(tdc.values())
                #addF = []
                #for i in range(len(f)):
                #    if type(f[i]) is str:
                #        addF.append(f[i])
                #c = Counter(addF)
                #tempDict = c.most_common(1)[0][0]
                tempDict = majority_label
                break
        classified.append(tempDict)
    return classified

def test_accuracy(classified_list, testMatrix):
    av_accuracy = 0
    sum_acc = 0
    accuracy_list = []
    num_test_ex = len(testMatrix)
    mat_col_last_ind = len(testMatrix[0])-1
    for i in range(num_test_ex):
        if testMatrix[i][mat_col_last_ind] == classified_list[i]:
            temp_acc = 1
        else:
            temp_acc = 0
        accuracy_list.append(temp_acc)
        sum_acc += temp_acc
    av_accuracy = sum_acc/num_test_ex*100
    return av_accuracy

def pre_processing_for_missing_attributes(trainMatrix, numAttrs, method):
    #method 0,3,any... : return trainingMatrix
    #method 1: Setting the missing feature as the majority feature value.
    #method 2: Setting the missing feature as the majority value of that label.
    numEx = len(trainMatrix)
    if method == 0:
        return trainMatrix
    elif method == 1:
        for col_ind in range(numAttrs):
            col_Value_list = elements_of_column(trainMatrix, col_ind)
            if '?' in col_Value_list:
                max_1st = max(set(col_Value_list), key=col_Value_list.count)
                if max_1st == '?':
                    c = Counter(col_Value_list)
                    major_attribute_Value = c.most_common(2)[1][0]
                else:
                    major_attribute_Value = max_1st
                for row_ind in range(numEx):
                    if trainMatrix[row_ind][col_ind] == '?':
                        trainMatrix[row_ind][col_ind] = major_attribute_Value
        return trainMatrix
    elif method == 2:
        last_col_labels = elements_of_column(trainMatrix, numAttrs)
        for col_ind in range(numAttrs):
            col_Value_list = elements_of_column(trainMatrix, col_ind)
            if '?' in col_Value_list:
                for row_ind in range(numEx):
                    if trainMatrix[row_ind][col_ind] == '?':
                        this_label = last_col_labels[row_ind]
                        temp_list = []
                        for new_row in range(numEx):
                            if col_Value_list[new_row] != '?':
                                if last_col_labels[new_row] == this_label:
                                    temp_list.append(col_Value_list[new_row])

                        count_attr_list = Counter(temp_list)
                        new_attribute = count_attr_list.most_common()[0][0]
                        trainMatrix[row_ind][col_ind] = new_attribute
        return trainMatrix
    elif method == 3:
        return trainMatrix
    else:
        return trainMatrix

def create_ID3_tree(Sdata, numAttributes, depth, to_be_visited, majority_label, limiting_depth):
    last_col = elements_of_column(Sdata, len(Sdata[0])-1)
    c_ = Counter(last_col)
    majority_label = (c_.most_common(1))[0][0]
    #take care of Questiomarks
    '''
    if numAttributes == 1:
        elements_list = treeFunctions.elements_of_column(Sdata, 0)
        sv_c = Counter(elements_list)
        if len(sv_c) == 1:
            return majority_label
    '''
    if numAttributes <= 0:
        return majority_label
    elif len(c_) == 1:
        label = list(c_)[0]
        return label
    else:
        best_info_gain, temp_A_index  = ID3_all_examples(Sdata,numAttributes)

        A_index = to_be_visited[temp_A_index]
        to_be_visited.pop(temp_A_index)
        #A_index = (temp_A_index)
        A_column = elements_of_column(Sdata, temp_A_index)
        dummy_prob, A_values, A_value_count = list_of_pi_s(A_column)
        tree = {A_index:{}}
        del_var = 0
        for val in range(len(A_values)):

            temp_depth = copy.deepcopy(depth)
            temp_depth += 1
            temp_to_be_visited = copy.deepcopy(to_be_visited)
            temp_Ddata = copy.deepcopy(Sdata)
            subSdata = trim_dataset(temp_Ddata , temp_A_index, A_values[val])
            numAttributes = len(subSdata[0])-1
            subtree = create_ID3_tree(subSdata, numAttributes, temp_depth, temp_to_be_visited,  majority_label, limiting_depth)
            if type(subtree) is str:
                tree[A_index][A_values[val]] = subtree
            elif (temp_depth > limiting_depth - 1):
                subtree = majority_label
                tree[A_index][A_values[val]] = subtree
            else:
                tree[A_index][A_values[val]] = subtree

            del_var = 0
    return tree

def ID3_implementation(Sdata, numAttributes, depth, to_be_visited, majority_label, limiting_depth, method):
    new_matrix = pre_processing_for_missing_attributes(Sdata, numAttributes, method)
    tree_structure = create_ID3_tree(new_matrix, numAttributes, depth, to_be_visited, majority_label, limiting_depth)
    return tree_structure

def cross_valid_Experiment(splitPath, limiting_depth, method):
    all_filenames = []
    for filename in os.listdir(splitPath):
        all_filenames.append(filename)
    accuracy_list = []
    for i in range(len(all_filenames)):
        temp_all_filenames = copy.deepcopy(all_filenames)
        poped_file = temp_all_filenames.pop(i)
        combined_data = []
        for files_ in temp_all_filenames:
            new_path = splitPath + '/' + files_
            inc_matrix, dummy, dummy = read_file(new_path)
            combined_data += inc_matrix
        newTestPath = splitPath + '/' + poped_file
        test_matr,  dummy, dummy = read_file(newTestPath)
        numTrainingAttributes = len(combined_data[0])-1
        depth_of_tree = 0
        to_be_visited = range(numTrainingAttributes)
        last_col = elements_of_column(combined_data, len(combined_data[0]) - 1)
        list_of_labels = list(Counter(last_col))
        majority_label = (Counter(last_col)).most_common(1)[0][0]
        tree = ID3_implementation(combined_data, numTrainingAttributes, depth_of_tree, to_be_visited, majority_label, limiting_depth , method)
        classified_list = classify_test(tree, test_matr, list_of_labels, majority_label)
        accuracy_ = test_accuracy(classified_list, test_matr)
        accuracy_list.append(accuracy_)
    av_accuracy = sum(accuracy_list) / len(all_filenames)
    std_deviation = np.std(accuracy_list)
    return av_accuracy, std_deviation