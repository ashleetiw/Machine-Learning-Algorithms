import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    
    transform_data=np.zeros((features.shape[0],features.shape[1]))
    # print(transform_data)
    for i in range(len(features)):
        mag=np.sum(features[i]**2)
        ang=np.arctan2(features[i][1],features[i][0])
        transform_data[i][0]=np.sqrt(mag)
        transform_data[i][1]=ang

    # print(transform_data)
    return transform_data

    # print(features[0])
    # print(features[0]**2)
    # print(np.sum(features[0]**2))
    

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.


        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        begin initialize self.weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using self.weights
                    then self.weights = self.weights + example * label_for_example
            return self.weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """

        
        n_dim=len(features[1])
        eg_len=len(targets)
        self.weights = np.random.rand(eg_len,n_dim+1)
        arr_1 = np.ones((eg_len,1))
        X=np.hstack((arr_1, features))

        
        for _ in range(self.max_iterations):
            c=0
            for index in range(len(targets)):
                if np.sign(np.dot(self.weights[index].T,X[index])) != targets[index]:
        # #             print(np.dot(W.T,(features))[index],targets[index])
                        self.weights[index] = self.weights[index] + X[index] * targets[index] 
                else:
                    c+=1
            
            if c==len(targets):
                break
        

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        # """
        predictions=[]
        arr_1 = np.ones((len(features),1))
        X=np.hstack((arr_1, features))

        for i in range(len(features)):
            result= np.sign(np.dot(self.weights[i].T,X[i]))
            predictions.append(result)

        
        return np.array(predictions)

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """

        plt.figure()
        plt.scatter(features,targets,color='blue', label='Data')
        features_temp = np.copy(features)
        features_temp.shape = (100,)
        features_temp = np.sort(features_temp)
        print(features_temp)
        features_temp.shape = (100,1)
        predicted = self.predict(features_temp)
        
        plt.plot(features_temp,predicted, color='green' , label='predicted')
        plt.axis([-1.5, 1.5, min(targets)-5,max(targets)+5])
        plt.title('X vs Y')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc="best")
        plt.savefig("result.png") 