import numpy as np 
from .distances import euclidean_distances, manhattan_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure=distance_measure
        self.aggregator=aggregator


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        self.features=features
        self.targets=targets
        

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        #  get distances of test points with each training point 

        # features: test data
        # self.features : train data 
        if(self.distance_measure=="manhattan"):
            dist = manhattan_distances(features,self.features)
        else:
            dist = euclidean_distances(features,self.features)  

        # get top K nearest points
        nearest_pts=[]
        for d in dist:  # for each row( test example)
            sorted_index = np.argsort(d,axis=0)  # sorts data and stores their indexes
            # print(d)
            # print(sorted_index)
            arr = [] 
            if (ignore_first==False):  #case 1 when ignore_first is False that means take all K closest points
                for k in range(self.n_neighbors):   # going for top K closest indexes
                    arr.append(sorted_index[k])
                nearest_pts.append(arr)
            else:
                 #  case 2 when ignore_first is True that means take all 1:K+1 closest points
                for k in range(1,self.n_neighbors+1):   # going for top K closest indexes
                    arr.append(sorted_index[k])
                nearest_pts.append(arr)
        
        # print(nearest_pts)

      
        
        # ######################   make prediction ########
        # Remember:nearest_pts is a 2d array with size nof exampleX K neighbours indexes
        
        predict=[]
        for i in nearest_pts: # i is each example
            label=[]
            for index in i: # index is each neighbour of point i 
                label.append(list(self.targets[index])) # gets class labels 
        
        
            label=np.array(label)
            result=[]
            for i in range(label.shape[1]):
                col = label[:,i]
                result.append(self.decide_aggregator(col))
            predict.append(result)

        return np.array(predict)

    def decide_aggregator(self,col):
        if self.aggregator=="mean":
            return np.mean(col)
        elif self.aggregator=="median":
            return np.median(col)
        else:
        
            vals,counts = np.unique(col, return_counts=True)
            index = np.argmax(counts)   
            return vals[index]






        