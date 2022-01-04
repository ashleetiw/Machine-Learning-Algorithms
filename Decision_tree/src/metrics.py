import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    confusion_matrix= np.zeros((2, 2))

    for i in range(len(predictions)):
        if actual[i]==predictions[i]:
            if (actual[i]):  #true_positives
                confusion_matrix[1][1] += 1
            else:             #true_negatives
                confusion_matrix[0][0] += 1
        else:
            if (actual[i]):  #false_negatives
                confusion_matrix[1][0] += 1
            else:             #false_positives
                confusion_matrix[0][1] += 1
            
    return confusion_matrix
    

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    matrix=confusion_matrix(actual, predictions)
    total_predicted = np.sum(matrix)
    total_correctpredicted = matrix[0][0]+matrix[1][1]
    accuracy=total_correctpredicted/float(total_predicted) 
    return accuracy
    

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    # precs[i] = tp/(fp+tp) if fp+tp > 0 else np.nan
    # recs[i] = tp/(fn+tp) if fn+tp > 0 else np.nan
    matrix=confusion_matrix(actual,predictions)
    
    tp=matrix[1][1]
    fp=matrix[0][1]
    fn=matrix[1][0]

    precision= tp/float(fp+tp) if  fp+tp > 0 else np.nan
    recall= tp/float(fn+tp) if fn+tp > 0 else np.nan

    return precision,recall


def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    p,r=precision_and_recall(actual,predictions)
    f1= 2.0*(p*r)/(p+r) if p+r >0 else np.nan
    return f1

