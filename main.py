from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import random
import copy
import sys

# df = pd.read_table('CS170_small_special_testdata__95.txt', sep=r'\s{2,}', header=None, engine='python')
# df = pd.read_table('CS170_small_special_testdata__95.txt', sep=r'\s+', header=None, engine='python')
# print(df)


def feature_search(data):
    numRows = data.shape[0]
    numCols = data.shape[1]

    current_set_of_features = set()
    
    globalBestAccuracy = 0
    globalBestSetOfFeatures = set()

    print('Beginning search.')
    for i in range(0, numCols):
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(1, numCols):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                print('\tConsidering feature {}, accuracy is {}%'.format(k, accuracy))

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k               

        # Needed because last iteration feature_to_add_at_this_level will be 0 and throw off set & accuracy
        if feature_to_add_at_this_level != 0:
            current_set_of_features.add(feature_to_add_at_this_level)
            print('Feature set {} was best, accuracy is {}%'.format(current_set_of_features, best_so_far_accuracy))
            
            if best_so_far_accuracy > globalBestAccuracy:
                globalBestAccuracy = best_so_far_accuracy
                globalBestSetOfFeatures.add(feature_to_add_at_this_level)
    
    print('\nFinished search!! The best feature subset is {}, which has an accuracy of {}%'.format(globalBestSetOfFeatures, globalBestAccuracy))


def backward_elimination_search(data):
    numRows = data.shape[0]
    numCols = data.shape[1]

    current_set_of_features = set()
    for feature in range(1, numCols):
        current_set_of_features.add(feature)
    
    globalBestAccuracy = 0
    globalBestSetOfFeatures = copy.deepcopy(current_set_of_features)

    print('Beginning search.')
    for i in range(0, numCols):
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(1, numCols):
            if k in current_set_of_features:
                temp_set_of_features = copy.deepcopy(current_set_of_features)
                temp_set_of_features.remove(k)
                accuracy = leave_one_out_cross_validation(data, temp_set_of_features, np.inf)
                print('\tConsidering removing feature {}, accuracy is {}%'.format(k, accuracy))

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k               

        # Needed because last iteration feature_to_remove_at_this_level will be 0 and throw off set & accuracy
        if feature_to_remove_at_this_level != 0:
            current_set_of_features.remove(feature_to_remove_at_this_level)
            print('Removed feature {}. Feature set is now {}, accuracy is {}%'.format(feature_to_remove_at_this_level, current_set_of_features, best_so_far_accuracy))
            
            if best_so_far_accuracy > globalBestAccuracy:
                globalBestAccuracy = best_so_far_accuracy
                globalBestSetOfFeatures.remove(feature_to_remove_at_this_level)
    
    print('\nFinished search!! The best feature subset is {}, which has an accuracy of {}%'.format(globalBestSetOfFeatures, globalBestAccuracy))


def leave_one_out_cross_validation(data, current_set, feature_to_add):
    numRows = data.shape[0]
    numCols = data.shape[1]

    newData = copy.deepcopy(data)

    number_correctly_classified = 0

    # Sets features we are not using to 0
    for feature in range(1, numCols):
        if feature not in current_set and feature != feature_to_add:
            newData[ : , feature] = 0

    for i in range(0, numRows):
        object_to_classify = newData[i, 1:]
        label_object_to_classify = newData[i, 0]

        # print('Looping over i, at the {} location'.format(i+1))
        # print('The {}th object is in class {}'.format(i+1, label_object_to_classify))
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf

        for k in range(0, numRows):
            # print('Ask if {} is nearest neighbor with {}'.format(i, k))
            if k != i:
                distance = np.sqrt(sum((object_to_classify - newData[k, 1:])**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = newData[nearest_neighbor_location, 0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    return round((number_correctly_classified / numRows) * 100, 1)

def main():
    print('Welcome to Isaac\'s Feature Selection Algorithm')
    file = input('Type the name of the file to test: ')
    data = np.loadtxt(file)

    print('Choose a search algorithm.\n\t1) Forward Selection\n\t2) Backward Elimination')
    algorithmChoice = input()

    print ('\n\nThis dataset has {} features (not including the class attribute), with {} instances.\n'.format(data.shape[1] - 1, data.shape[0]))

    if algorithmChoice == '1':
        feature_search(data)
    elif algorithmChoice == '2':
        backward_elimination_search(data)

main()