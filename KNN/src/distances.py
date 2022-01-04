import numpy as np 

def e_dist(vec1,vec2):
    # dist=vec1-vec2
    # res=np.dot(dist.T,dist) # sum of squares
    # return np.sqrt(res)

    ans=np.linalg.norm(vec1-vec2, ord = None)
    return ans

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    # print(X.shape[0],X.shape[1])
    # print(Y.shape[0],Y.shape[1])     

    # note shape[1] same for both matrices 
    M=X.shape[0]
    K=X.shape[1]
    N=Y.shape[0]
    
    D = np.zeros((M,N))
     #calculate distance 
    for i in range(M):
        point1= X[i]
        for j in range(N):
            point2 = Y[j]
            dist = e_dist(point1,point2)
            # storing euclidean distance bettwen point at ith index and j the index 
            D[i][j] = dist
    return D 


def m_dist(vec1,vec2):
    ans=np.linalg.norm(vec1-vec2, ord =1)
    return ans

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    M = X.shape[0]
    N = Y.shape[0]
    D = np.zeros((M,N))
    #calculate distance 
    for i in range(M):
        point1 = X[i]
        for j in range(N):
            point2 = Y[j]
            dist = m_dist(point1,point2)
            D[i][j] = dist
    return D
    