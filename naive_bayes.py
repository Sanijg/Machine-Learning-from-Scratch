"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np
import math

def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    alpha = params['alpha']
    all_labels = np.unique(train_labels)
    d, n = train_data.shape
    num_classes = all_labels.size
    

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES
    prob_ynx= []
    class_count, class_prob = class_probability(train_labels, alpha) #Helper function to find the class probability
    
    #Here onward, conditional probability calculation is doneis done
    
    for c in range(num_classes): #pure classes
        temp = np.zeros(n)
        for j in range(n): #total number of examples
            if train_labels[j]== all_labels[c]:
                temp[j]=1
        y_n_x_count = []
        for k in range(d): #total attribute
            single= np.sum(train_data[k,:]*temp) #
            y_n_x_count.append(single)
        prob_ynx.append((y_n_x_count+alpha*np.ones(d))/((class_count[c]+2*alpha)*np.ones(d)))
    prob_ynx= np.asarray(prob_ynx)
        
    '''
    two arrays are returned:
        shapes: (20,)
                (20,5000)
    '''
    return class_prob,prob_ynx

def class_probability(labels, alpha):
    all_labels = np.unique(labels)
    n= len(labels)
    num_classes = len(all_labels)
    class_count = np.zeros(num_classes)
    class_prob = np.zeros(num_classes)
    
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob[c] = (class_count[c]+ alpha)/(n+20*alpha)
    return class_count, class_prob
            
    
                
       
def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA
    class_prob,prob_ynx = model
    class_prob=np.log10(class_prob)
    prob_not_ynx = np.log10(1-prob_ynx)
    prob_ynx= np.log10(prob_ynx)
    d, n = data.shape
    num_classes = len(class_prob)
    prediction = []
    for i in range(n):    #Because I need to make n prediction
        temp =[]
        for j in range(num_classes): #Because there are 20 different classes
            s = class_prob[j]
            s+= np.dot(prob_ynx[j,:],data[:,i])
            not_vector= np.logical_not(data[:,i])  #Because to incorporate 'False' cases
            s+= np.dot(prob_not_ynx[j,:],not_vector)
            temp.append(s)
        prediction.append((np.asarray(temp).argmax()))
    return prediction

""" Accuracies:
            Naive Bayes training accuracy: 0.7555968
            Naive Bayes testing accuracy: 0.621985
""" 