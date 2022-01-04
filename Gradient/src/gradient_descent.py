import numpy as np
from your_code import HingeLoss, SquaredLoss
from your_code import L1Regularization, L2Regularization
from your_code import metrics

class GradientDescent:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate
        self.q1 = False
        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        
        
        D=features.shape[-1]
        N=features.shape[0]
        N_ones=np.ones((features.shape[0],1))
        batch_counter = 0 
        # add a column with ones at the end of features
        features0=np.append(features,N_ones,axis=1)
        # np.random.seed(0)
        self.weights=np.random.uniform(-0.1,0.1,features.shape[-1]+1) 
        

        Los_old=1000  
        for i in range(max_iter):
            if batch_size is None :
                features1=features0
                targets1 = targets

            else :   # select batch_size random features
                if((batch_size*i)%N==0): # shuffle main list if we are at epoxi
                    randomize = np.arange(N) 
                    np.random.shuffle(randomize)
                    features0 = features0[randomize]
                    targets = targets[randomize]
                    batch_counter =0
                #taka features of batch_size from batch_size*i --> batch_size*(i+1)
                features1=features0[batch_size*batch_counter:batch_size*(batch_counter+1)]
                targets1 = targets[batch_size*batch_counter:batch_size*(batch_counter+1)]
                batch_counter +=1

                    
            grd=self.loss.backward(features1,self.weights,targets1)
            self.weights=self.weights-self.learning_rate*grd
            Los=self.loss.forward(features1,self.weights,targets1)

            # for free response questions keep track of data 
            if(self.q1):
                if batch_size==None :
                    # print("iteration %d" %i)
                    prediction = self.predict(features1)
                    accuracy = metrics.accuracy(ground_truth=targets1,predictions=prediction)
                    self.loosPoints.append(Los)
                    self.accuracyPoints.append(accuracy)
                elif((batch_size*i)%N==0):
                    # print("epoxi %d %d" %(i,i*batch_size))
                    prediction = self.predict(features1)
                    accuracy = metrics.accuracy(ground_truth=targets1,predictions=prediction)
                    self.loosPoints.append(Los)
                    self.accuracyPoints.append(accuracy)

            diff=abs(Los_old-Los)
            # print("i=",i,W,diff)
            Los_old=Los
            if diff < 0.0001:
                break
            
        self.last_iter = i
        self.last_loss = Los

    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        y_predict=self.confidence(features)
        predict=y_predict
        predict[predict<0]=-1
        predict[predict>=0]=1
        
        return predict

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        N=features.shape[0]
        N_ones=np.ones((N,1))
        if(features.shape[1] != self.weights.shape[0]):
            features1=np.append(features,N_ones,axis=1)
        else:
            features1 = features
        y_predict=np.matmul(features1,self.weights)
        
        return y_predict

    
    def question1(self):
        self.q1 = True
        self.loosPoints = []
        self.accuracyPoints = []
