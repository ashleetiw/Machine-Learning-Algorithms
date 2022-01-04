import numpy as np
import random
import math

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        self.n_samples,self.n_features = features.shape
        self.features = features

        self.rand_centroid()

        # print(self.means)
        old_means = self.means.copy()
        i = 0
        while(True):
        # for i in range(10):
            self.group_by_centroids()
            i = i + 1
            sub = old_means - self.means
            res = sum(sum(sub))
            if(res==0):
                break
            else:
                old_means = self.means.copy()
               
            
        self.means = np.array(self.means)
          

    def group_by_centroids(self):
        feature_index = 0 
        list_centroid = []
        for i in range(self.n_clusters):
            list_centroid.append([])


        for i in self.features:
            index = self.find_nearest_centroid(i)
            list_centroid[index].append(feature_index)
            feature_index += 1

        self.update_centroids(list_centroid)        


    def find_nearest_centroid(self,sample):
        dist = []
        for centroid in self.means:
            distance = self.distance(sample,centroid)
            dist.append(distance)
        minDist = min(dist)
        index = dist.index(minDist)
        return index       

    def update_centroids(self,list_centroid):
        
        for id in range(self.n_clusters):
            
            new_centroid = np.zeros(self.n_features)
            list_index = list_centroid[id]
            for index in list_index:
                new_centroid = new_centroid + self.features[index]
            new_centroid = (1.0*new_centroid)/len(list_index)
            self.means[id] = new_centroid


    def distance(self,sample,centroid):
        dist = centroid - sample
        dist = dist*dist
        res = np.sum(np.sum(dist))
        res = math.sqrt(res)
        return res

    
    def rand_centroid(self):
        self.means = []
        for i in range(self.n_clusters):
            self.means = random.sample(list(self.features),self.n_clusters)
            self.means = np.array(self.means)
        
        self.means = np.array(self.means)



    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        prediction = []
        for i in features:
            index = self.find_nearest_centroid(i)
            prediction.append(index)
        return prediction