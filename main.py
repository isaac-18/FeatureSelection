from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import random
import copy

# df = pd.read_table('CS170_small_special_testdata__95.txt', sep=r'\s{2,}', header=None, engine='python')
# df = pd.read_table('CS170_small_special_testdata__95.txt', sep=r'\s+', header=None, engine='python')
# print(df)


def feature_search(data):
    numRows = data.shape[0]
    numCols = data.shape[1]

    current_set_of_features = set()

    print('Beginning search.')
    for i in range(0, numCols):
        # print('On the {}th level of the search tree.'.format(i))
        # feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(1, numCols):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                print('\tConsidering feature {}, accuracy is {}'.format(k, accuracy))

                if accuracy > best_so_far_accuracy:
                    # print('Best so far accuracy: {}'.format(accuracy))
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        current_set_of_features.add(feature_to_add_at_this_level)
        # print('On level {} I added feature {} to current set'.format(i, feature_to_add_at_this_level))
        # print(best_so_far_accuracy)
        # print(current_set_of_features)
        print('Feature set {} was best, accuracy is {}'.format(current_set_of_features, best_so_far_accuracy))

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

    return number_correctly_classified / numRows

def main():
    data = np.loadtxt('CS170_small_special_testdata__95.txt')
    # print(data.shape)
    feature_search(data)
    # leave_one_out_cross_validation(data, 0, 0)

    # print(data)
    # current_set = {2, 9}
    # feature_to_add = 1

    # for feature in range(1, 11):
    #     if feature not in current_set and feature != feature_to_add:
    #         data[ : , feature] = 0

    # print('=====================================')
    # print(data)
    # print('=====================================')
    # print(data[ : , 10])

main()