import numpy as np
from sklearn.naive_bayes import GaussianNB
from ucimlrepo import fetch_ucirepo

def _groundTruthMatrix(y_true, labels):
    """
    Return the ground truth matrix from a given array of true values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    labels : list with all the classes of the dataset.

    Returns
    ----------
    groundTruthM : numpy array(n_samples, n_classes).
        Returns the ground truth matrix.
    """
    # Size of the test
    size = y_true.shape[0]

    # Initializes some utility arrays
    groundTruthM = np.zeros((size, len(labels)))
    index = np.array([])

    # Makes a binary array from y_true
    for i in range(size):
        class_y = y_true[i]
        for j in range(len(labels)):
            if(labels[j] == class_y):
                index = np.append(index, np.where(labels == labels[j])[0])

    # Turns the binary array into a binary matrix
    groundTruthM[np.arange(size), index.astype(int)] = 1

    return groundTruthM

def prob_confusion_matrix(y_true, y_prob, labels=None, abs_tolerance=1e-8):
    """
    Calculate the probabilistic confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : list with all the classes of the dataset, default=None.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1.0, default = 1e-8.

    Returns
    ----------
    prob_conf_matrix : numpy array(n_classes, n_classes).
        Returns the probabilistic confusion matrix.
    """
    # Checks y_true data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        y_true = np.asarray(y_true)#.to_numpy()
        y_prob = np.asarray(y_prob)
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples.") 
    elif len(np.unique(y_true)) != y_prob.shape[1]:
        raise ValueError("'y_true' and 'y_score' have different number of classes.")
    elif not np.allclose(1, y_prob.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities and they should sum up to 1.0 over classes."
        )
    
    # Classes of the test
    if labels == None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    # Transforms y_test into a proper ground truth matrix
    groundTruthM = _groundTruthMatrix(y_true, labels)

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = np.dot(np.transpose(groundTruthM), y_prob)

    return prob_conf_matrix

def prob_accuracy(y_true, y_prob, labels=None):
    """
    Calculate the probabilistic accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : list with all the classes of the dataset, default=None.

    Returns
    ----------
    prob_acc : float.
        Returns the probabilistic accuracy.
    """
    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_prob, labels)

    # Calculates the probabilistic accuracy
    prob_acc = np.sum(np.diag(prob_conf_matrix))/np.sum(prob_conf_matrix)

    return prob_acc

def certainty_matrix(y_true, y_prob, labels=None, abs_tolerance=1e-8):
    """
    Calculate the certainty and uncertainty matrices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : list with all the classes of the dataset, default=None.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1.0, default = 1e-8.

    Returns
    ----------
    V : numpy array(n_classes, n_classes).
        Returns the certainty matrix.

    U : numpy array(n_classes, n_classes).
        Returns the uncertainty matrix.
    """
    # Checks y_true data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        y_true = np.asarray(y_true)#.to_numpy()
        y_prob = np.asarray(y_prob)
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples.") 
    elif len(np.unique(y_true)) != y_prob.shape[1]:
        raise ValueError("'y_true' and 'y_score' have different number of classes.")
    elif not np.allclose(1, y_prob.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities and they should sum up to 1.0 over classes."
        )
    
    # Classes of the test
    if labels == None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    # Transforms y_test into a proper ground truth matrix
    groundTruthM = _groundTruthMatrix(y_true, labels)

    # Initializes two empty matrix and one empty array
    certM = np.zeros((y_prob.shape[0], y_prob.shape[1]))
    uncertM = np.zeros((y_prob.shape[0], y_prob.shape[1]))
    classes = np.zeros((y_prob.shape[1]))

    # Fills the empty array with a different number for each different class
    for i in range(1, y_prob.shape[1]):
        classes[i] = i

    # Saves the index for the most likely prediction from the probabilistic prediction matrix
    certIndex = np.argmax(y_prob, axis=1)

    # Saves the probabilistic predictions into the certainty and uncertainty matrix
    for i in range(y_prob.shape[0]):
        uncertIndex = np.delete(classes.astype(int), certIndex[i])
        certM[i][certIndex[i]] = np.max(y_prob[i])
        for j in range(len(uncertIndex)):
            uncertM[i][uncertIndex[j]] = y_prob[i][uncertIndex[j]]

    # Computes the certainty matrix and the uncertainty matrix
    V = np.dot(np.transpose(groundTruthM), certM)
    U = np.dot(np.transpose(groundTruthM), uncertM)

    return V, U

def certainty_weights(y_true, y_prob, labels=None):
    """
    Calculate the lambda values for the certainty and uncertainty matrices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : list with all the classes of the dataset, default=None.

    Returns
    ----------
    lambda_V : float.
        Returns the lambda value for the certainty matrix.

    lambda_U : float.
        Returns the lambda value for the uncertainty matrix.
    """
    # Computes the probabilistic confusion matrix 
    prob_conf_matrix = prob_confusion_matrix(y_true, y_prob, labels)

    # Computes the certainty and uncertainty matrices 
    V, U = certainty_matrix(y_true, y_prob, labels)

    # Calculates lambda values for the certainty and uncertainty matrices
    lambda_V = np.sum(V)/np.sum(prob_conf_matrix)
    lambda_U = np.sum(U)/np.sum(prob_conf_matrix)

    return lambda_V, lambda_U

def certainty_accuracy(y_true, y_prob, labels=None):
    """
    Calculate the probabilistic accuracy for the certainty and uncertainty matrices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : list with all the classes of the dataset, default=None.

    Returns
    ----------
    V_acc : float.
        Returns the probabilistic accuracy for the certainty matrix.

    U_acc : float.
        Returns the probabilistic accuracy for the uncertainty matrix.
    """
    # Computes the certainty and uncertainty matrices 
    V, U = certainty_matrix(y_true, y_prob, labels)

    # Calculates the probabilistic accuracy
    V_acc = np.sum(np.diag(V))/np.sum(V)
    U_acc = np.sum(np.diag(U))/np.sum(U)

    return V_acc, U_acc

if __name__ == "__main__":
    # Loads the dataset
    iris  = fetch_ucirepo(id=53) 
    X, y = iris.data.features, iris.data.targets.squeeze()

    classes = np.unique(y)
    print(classes)

    # Training and predict
    model = GaussianNB().fit(X, y)
    result = model.predict_proba(X)
    
    # Calculates the probabilistic confusion matrix and the probabilistic accuracy
    prob_conf_matrix = prob_confusion_matrix(y, result)
    prob_acc = prob_accuracy(y, result)

    print(np.round(prob_conf_matrix,3))
    print(f"Acc*:{np.round(prob_acc,5)}\n")


    # Calculates the certainty and uncertainty confusion matrix, their probabilistic accuracy and their lambda values
    V, U = certainty_matrix(y, result)
    V_acc, U_acc = certainty_accuracy(y, result)
    lambda_V, lambda_U = certainty_weights(y, result)

    print(np.round(V,3))
    print(f"Acc_V*:{np.round(V_acc,5)}, lambda_V:{np.round(lambda_V,5)}\n")
    print(np.round(U,3))
    print(f"Acc_U*:{np.round(U_acc,5)}, lambda_U:{np.round(lambda_U,5)}")