"""This module includes methods for training and predicting using decision trees."""
import numpy as np

def calculate_information_gain(data, labels):
    """
    Computes the information gain on label probability for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape

    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            full_entropy -= class_prob * np.log(class_prob)
    #print("Full entropy is %d\n" % full_entropy)
 
    gain = full_entropy * np.ones(d)

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    # EPr(X=xj) is handled here:
    num_x = data.astype(float).dot(np.ones(n))
    prob_x = num_x / n
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        #print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        #print(num_y)
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]] #Y=yj
        #print(data_with_label[1].shape)
        #print(data_with_label.shape)
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8)
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
     """Train a decision tree to classify data using the entropy decision criterion.
 
     :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
     :type train_data: ndarray
     :param train_labels: length n numpy vector with integer labels
     :type train_labels: array_like
     :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
     :type params: dict
     :return: dictionary encoding the learned decision tree
     :rtype: dict
     """
     max_depth = params['max_depth']
 
     labels = np.unique(train_labels)
     num_classes = labels.size

     model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
     return model
 
def class_counts(labels):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for one in labels:
        label = one
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def partition(data, labels, best_index):
    """Partitions a dataset.

    For each row in the dataset, check if it is True/False. If
    so, add it to 'data_true', otherwise, add it to 'data_false'.
    """
    d,n = data.shape
    data_true, data_false = [], []
    labels_true, labels_false = [], []
    for i in range(n): 
        if data[best_index,i]:
            data_true.append(data[:,i])
            labels_true.append(labels[i])
        else:
            data_false.append(data[:,i])
            labels_false.append(labels[i])
    return np.asarray(list(zip(*data_true))), np.asarray(labels_true), np.asarray(list(zip(*data_false))), np.asarray(labels_false)

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 word,
                 true_branch,
                 false_branch):
        self.word = word
        self.true_branch = true_branch
        self.false_branch = false_branch



def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    # TODO: INSERT YOUR CODE FOR LEARNING THE DECISION TREE STRUCTURE HERE
    # Rows are sorted in a manner where the top rows give higher InfoGain.
    
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    all_labels = np.unique(labels)
    num_classes = len(all_labels)
    
    gain = calculate_information_gain(data, labels)
    best_index = gain.argmax()  #Index to the best gain possible attribute
    
    # Base case: no further info gain or max. depth attained
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain[best_index] < 1e-04 or depth>=max_depth:  #e-04 is there to represent no appreciative info-gain situation
        return Leaf(labels)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    data_true, labels_true, data_false, labels_false = partition(data, labels, best_index)

    # Recursively build the true branch.
    true_branch = recursive_tree_train(data_true, labels_true, depth+1, max_depth, num_classes)

    # Recursively build the false branch.
    false_branch = recursive_tree_train(data_false, labels_false, depth+1, max_depth, num_classes)

    # Return a Word node.
    # This records the word to split on,
    # as well as the branches to follow
    # depending on the answer.
    return Decision_Node(best_index , true_branch, false_branch)



def decision_tree_predict(data, model):
     """Predict most likely label given computed decision tree in model.
     
     :param data: d x n ndarray of d binary features for n examples.
     :type data: ndarray
     :param model: learned decision tree model
     :type model: dict
     :return: length n numpy array of the predicted class labels
     :rtype: array_like
     """
     # TODO: INSERT YOUR CODE FOR COMPUTING THE DECISION TREE PREDICTIONS HERE
     d, n = data.shape
     label= []   #Initializing the prediction result list
     for c in range(n):
         temp=single_predict(data[:,c], model)
         label.append(temp)
     return label

def single_predict(data,model):
    # Base case: we've reached a leaf
    if isinstance(model, Leaf):
         temp = model.predictions
         return max(temp, key=temp.get)
        
    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if data[model.word]:
        return single_predict(data, model.true_branch)
    else:
        return single_predict(data, model.false_branch)
    
""" Accuracies:
            Decision tree training accuracy: 0.488775
            Decision tree testing accuracy: 0.342172
"""
