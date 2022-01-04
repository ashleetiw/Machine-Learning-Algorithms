import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value
        self.predict = None

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None
        self.attribute_names_dic = {}
        for i in range(len(self.attribute_names)):
            item = self.attribute_names[i]
            self.attribute_names_dic[item] = i

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        if(not self.tree):
            self.tree = Node()
            self.tree.branches.append(self.form_tree(features,targets,self.attribute_names_dic))
            self.visualize()


    def form_tree(self,features,targets,attributesDict):
        # Create a root node for the tree       
        # If all examples are positive, Return the single-node tree Root, with label = +.
        if(len(targets)==np.sum(targets)):
            return Node(attribute_name = "leaf",value=1)

        # If all examples are negative, Return the single-node tree Root, with label = -.        
        elif(np.sum(targets)==0):
            return Node(attribute_name = "leaf",value=0)

        # If number of predicting attributes is empty, then Return the single node tree Root,
        # with label = most common value of the target attribute in the examples.       
        elif(len(attributesDict)==0):
            return Node(attribute_name = "leaf",value=self.get_mode(targets))

        else:
            root_node=Node()
            Attribute,IG=self.best_IG(features,targets,attributesDict)
            index = attributesDict[Attribute]

            # Decision Tree attribute for Root = Attribute
            root_node.attribute_name=Attribute
            root_node.attribute_index= self.attribute_names.index(root_node.attribute_name)

            # pop the element 
            attributesDict_copy = attributesDict.copy()
            attributesDict_copy.pop(Attribute) 

            # split using the best attribute 
            pos,neg,postar,negtar=self.split_data(features, targets, index)
            pos=np.array(pos)
            postar=np.array(postar)
            neg=np.array(neg)
            negtar=np.array(negtar)
            
            left_child = self.form_tree(pos,postar,attributesDict_copy)
            right_child = self.form_tree(neg,negtar,attributesDict_copy)

            root_node.branches.append(left_child)
            root_node.branches.append(right_child)    
        
        return root_node


    def split_data(self, features, targets, index):
        
        pos  = []
        neg  = []
        postar = []
        negtar = []
        for i in range(features.shape[0]):
            if(features[i][index]==0):
                neg.append(features[i][:])
                negtar.append(targets[i])
            else:
                pos.append(features[i][:])
                postar.append(targets[i])

        return  pos,neg,postar,negtar

    def best_IG(self,features,targets,attributesDict):
        # initilaize with some number 
        ind=-1
        best_gain=-1000

        #  if found a bettern gain then subsitute value 
        for name,val in attributesDict.items():
            temp= information_gain(features,val,targets)
            if(temp>best_gain):
                best_gain=temp
                ind=val
        #  return the attribute with the best gain
        return self.attribute_names[ind],best_gain

    def get_mode(self,targets):
    
        if np.count_nonzero(targets == 1) > np.count_nonzero(targets == 0):
            return 1
        else:
            return 0


    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        self._check_input(features)
        predictions = np.zeros((features.shape[0]))
        i = 0
        for p in features:
            predict_val=self.predict_tree(self.tree,p)
            predictions[i] = predict_val
            i += 1

        return predictions

    def predict_tree(self,root,p):
        if root.branches == []:
            return root.value
        else:
            # print(p[root.attribute_index])
            if (p[root.attribute_index] == 0):
                return self.predict_tree(p, root.branches[0])
            else:
                return self.predict_tree(p, root.branches[1])


    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        # print("level, tab_level, tree.attribute_name, val is ")
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def calc_H(attributes):
    total=len(attributes)
    #  1 all positives 0 all negatives 
    # unqiue,p= np.unique(attributes, return_counts=True)
    p=np.sum(attributes)
    n=total-p

    if(p==0):
        pos = 0
    else:
        pos = -(p/total)*np.log2(p/total)
    if(n==0):
        neg = 0
    else:
        neg = -(n/total)*np.log2(n/total)
    return pos+neg


def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    root_entropy=calc_H(targets)

    # split the dataset for each attribute based on attribute index 
    PS=[]
    NS=[]
    total=len(targets)
    for i in range(len(features)):
        if(features[i][attribute_index]==1):
            PS.append(targets[i])
        else:
            NS.append(targets[i])

    IG=root_entropy-(len(PS)/total)*calc_H(PS)-(len(NS)/total)*calc_H(NS)
    # print('tgg',IG)
    return IG 


if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
