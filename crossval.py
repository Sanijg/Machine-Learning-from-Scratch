"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)

    d, n = all_data.shape

    indices = np.array(range(n), dtype=int)
    
    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length
    indices = indices.reshape((examples_per_fold, folds))
    
    models = []

    # TODO: INSERT YOUR CODE FOR CROSS VALIDATION HERE
    for i in range(folds):
        curr_indices = np.delete(indices, i, axis=1)
        curr_indices = curr_indices.flatten()
        curr_indices = curr_indices[curr_indices != -1]
        
        current_data = all_data[:, curr_indices]
        current_labels = all_labels[curr_indices]
        
        held_out_indices = indices[:, i]
        held_out_indices = held_out_indices.flatten()
        held_out_indices = held_out_indices[held_out_indices != -1]
        
        held_out_data = all_data[:, held_out_indices]
        held_out_labels = all_labels[held_out_indices]
        
        model = trainer(current_data, current_labels, params)
        models.append(model)
        
        nb_test_predictions = predictor(held_out_data, model)
        nb_test_accuracy = np.mean(nb_test_predictions == held_out_labels)
        scores[i] = nb_test_accuracy
        
    score = np.mean(scores)

    return score, models
