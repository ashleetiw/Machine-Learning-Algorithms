import numpy as np
from src import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        """
        This class implements a Gaussian Mixture Model updated using expectation
        maximization.

        A useful tutorial:
            https://campuspro-uploads.s3.us-west-2.amazonaws.com/63aa7cea-5e9c-4b62-96b7-8bbf3bc31b76/3a1d9101-8748-4e85-9830-4e45ffe1ca8d/EM%20derivations.pdf

        The EM algorithm for GMMs has two steps:

        1. Update posteriors (assignments to each Gaussian)
        2. Update Gaussian parameters (means, variances, and priors for each Gaussian)

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you break these two steps apart into separate
        functions. We have provided a template for you to put your code in.

        Use only numpy to implement this algorithm.

        This function MUST, after running 'fit', have variables named 'means' and
        'covariances' in order to pass the test cases. These variables are checked by the
        test cases to make sure you have recovered cluster parameters accurately.

        The fit and predict functions are implemented for you. To complete the implementation,
        you must implement:
            - _e_step
            - _m_step
            - _log_likelihood

        Args:
            n_clusters (int): Number of Gaussians to cluster the given data into.
            covariance_type (str): Either 'spherical', 'diagonal'. Determines the
                covariance type for the Gaussians in the mixture model.

        """
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def fit(self, features):
        """
        Fit GMM to the given data using `self.n_clusters` number of Gaussians.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means, covariances, and mixing weights - internally)
        """
        # 1. Use your KMeans implementation to initialize the means of the GMM.
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        # 2. Initialize the covariance matrix and the mixing weights
        self.covariances = self._init_covariance(features.shape[-1])

        # 3. Initialize the mixing weights
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        # 4. Compute log_likelihood under initial random covariance and KMeans means.
        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)


        # 5. While the log_likelihood is increasing significantly, or max_iterations has
        # not been reached, continue EM until convergence.
        n_iter = 0
        while log_likelihood - prev_log_likelihood > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)

            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )
        
        
            log_likelihood = self._overall_log_likelihood(features)
            n_iter += 1
        # print("covarience type\n",self.covariance.shape[0],self.covariance.shape)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict the label
        of each sample (e.g. the index of the Gaussian with the highest posterior for that
        sample).

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted assigment to each cluster for each sample,
                of size (n_samples,). Each element is which cluster that sample belongs to.
        """
        posteriors = self._e_step(features)
        # print(np.argmax(posteriors, axis=-1))
        return np.argmax(posteriors, axis=0)[0]

    def _e_step(self, features):
        """
        The expectation step in Expectation-Maximization. Given the current class member
        variables self.mean, self.covariance, and self.mixing_weights:
            1. Calculate the log_likelihood of each point under each Gaussian.
            2. Calculate the posterior probability for each point under each Gaussian
            3. Return the posterior probability (assignments).
        
        This function should call your implementation of of _posterior which in turn calls 
        _log_likelihood (which should call
        multvariate_normal.logpdf). This should use the Gaussian parameter contained in
        self.means, self.covariance, and self.mixing_weights

        Arguments:
            features {np.ndarray} -- Features to apply means, covariance, and mixing_weights
                to.

        Returns:
            np.ndarray -- Posterior probabilities to each Gaussian (shape is
                (features.shape[0], self.n_clusters))
        """
        posterior = [
            self._posterior(features, j) for j in range(self.n_clusters)
        ]

        return np.array(posterior)

    def _m_step(self, features, assignments):
        """
        Maximization step in Expectation-Maximization. Given the current features and
        assignments, update self.means, self.covariances, and self.mixing_weights. Here,
        you implement the update equations for the means, covariances, and mixing weights.
            1. Update the means with the mu_j update in Slide 24.
            2. Update the mixing_weights with the w_j update in Slide 24
            3. Update the covariance matrix with the sigma_j update in Slide 24.

        Slide 24 is in these slides: 
            https://github.com/NUCS349/nucs349.github.io/blob/master/lectures/eecs349_gaussian_mixture_models.pdf

        NOTE: When updating the parameters of the Gaussian you always use the output of
        the E step taken before this M step (e.g. update the means, mixing_weights, and covariances 
        simultaneously).

        Arguments:
            features {np.ndarray} -- Features to update means and covariances, given the
                current assignments.
            assignments {np.ndarray} -- Soft assignments of each point to one of the cluster,
                given by _e_step.

        Returns:
            means -- Updated means
            covariances -- Updated covariances
            mixing_weights -- Updated mixing weights
        """

        #initialize
        # means
        mixing_weights = np.empty(self.mixing_weights.shape, dtype=object) 
        means = np.empty(self.means.shape, dtype=object) 
        covariances = np.zeros(self.covariances.shape)
        # print(covariances.shape)
        Gamma = np.zeros(self.n_clusters)
        N = assignments[0].shape[-1]*1.0
        D = features.shape[-1]

        for j in range(self.n_clusters):
            gamma = assignments[j]
            Gamma[j] = np.sum(gamma)


        for k in range(self.n_clusters):
            gamma = assignments[k]
            mixing_weights[k] = Gamma[k]/N
            means[k] = np.sum(gamma.T*features,axis=0)/Gamma[k]
            
            if self.covariance_type == 'spherical':
                for i in range(features.shape[0]):
                    covariances[k] += gamma[0][i]*np.dot((features[i] - self.means[k]),(features[i] - self.means[k]).T)
                covariances[k] = covariances[k]/(Gamma[k]*D*1.0)
            elif self.covariance_type == 'diagonal':
                cov = np.zeros([D,D])
                for i in range(features.shape[0]):
                    mk = features[i] - self.means[k]
                    mk = np.array([mk])
                    cov = cov + gamma[0][i]*np.matmul(mk.T,mk)
                    # print(cov.shape)
                    # print(cov)
                    # covariances[k]=cov
                for i in range(D):
                    covariances[k][i] = cov[i][i]
                covariances[k] = covariances[k]/(Gamma[k])


            

        return means,covariances,mixing_weights




    def _init_covariance(self, n_features):
        """
        Initialize the covariance matrix given the covariance_type (spherical or
        diagonal). If spherical, each feature is treated the same (has equal covariance).
        If diagonal, each feature is treated independently (n_features covariances).

        Arguments:
            n_features {int} -- Number of features in the data for clustering

        Returns:
            [np.ndarray] -- Initial covariances (use np.random.rand)
        """
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def _log_likelihood(self, features, k_idx):
        """
        Compute the likelihood of the features given the index of the Gaussian
        in the mixture model. This function compute the log multivariate_normal
        distribution for features given the means and covariance of the ```k_idx```th
        Gaussian. To do this, you can use the function:

            scipy.stats.multivariate_normal.logpdf

        Read the documentation of this function to understand how it is used here:

            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html

        Once the raw likelihood is computed, incorporate the mixing_weights for the Gaussian
        via:

            log(mixing_weight) + logpdf

        Where logpdf is the output of multivariate_normal.

        Arguments:
            features {np.ndarray} -- Features to compute multivariate_normal distribution
                on.
            k_idx {int} -- Which Gaussian to use (e.g. use self.means[k_idx], 
                self.covariances[k_idx], self.mixing_weights[k_idx]).

        Returns:
            np.ndarray -- log likelihoods of each feature given a Gaussian.
        """
        # cov = self.covariance_type[k_idx]
        # if self.covariance_type == 'spherical':
        #     cov = self.covariances[k_idx]
        # elif self.covariance_type == 'diagonal':
        #     D = self.covariances[k_idx].shape[-1]
        #     cov = np.zeros(D,D)
        #     for i in range(D):
        #         cov[i][i] = self.covariances[k_idx][i]

        return np.log(self.mixing_weights[k_idx]) + multivariate_normal.logpdf(features,self.means[k_idx],self.covariances[k_idx])

    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)

    def _posterior(self, features, k):
        """
        Computes the posteriors given the log likelihoods for the GMM. Computes
        the posteriors for one of the Gaussians. To get all the posteriors, you have
        to iterate over this function. This function is implemented for you because the
        numerical issues can be tricky. We use the logsumexp trick to make it work (see
        below).

        Arguments:
            features {np.ndarray} -- Numpy array containing data (n_samples, n_features).
            k {int} -- Index of which Gaussian to compute posteriors for.

        Returns:
            np.ndarray -- Posterior probabilities for the selected Gaussian k, of size
                (n_samples,).
        """
        num = self._log_likelihood(features, k)
        denom = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])

        # Below is a useful function for safely computing large exponentials. It's a common
  

        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors