import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        np.random.seed()

        if (self.k ==0 ):
            raise ValueError("Cannot provide K = 0")

        if (self.k > 200):
            raise ValueError("Cannot provide K bigger than 200")
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if (mat.shape[0] < self.k):
            raise ValueError("Cannot provide K bigger than observations")
    
        ## pick a random number to initialize the clusters
        ran = np.random.choice(len(mat), self.k, replace=False,)
        ## This are the first clusters
        centroids = mat[ran, :]

        ### initialize the errors to a large value
        error = np.ones(self.k)* np.inf * -1  ## well use them in _get_error
        
        while (self.max_iter >= 0):
            distance=(cdist(mat, centroids,metric=self.metric)) ## we calcuate the distance of the matrix points to any of the clusters
            points_classification=[]
            for i in distance:
                points_classification.append(np.argmin(i)) # Get the index with the minimun distance
            points_classification=np.array(points_classification) ## convert to a np array

            centroids = self._get_centroids(mat,points_classification) ## We call the get centroids with our matrix and the label of the points
            ## keep trak of the last error
            prev_error = error 
            # get the MSE from the centroids to the points
            error = self._get_error(centroids,mat,points_classification)
            # if the error is less than the tolerance, exit and return
            if (np.all((error - prev_error)< self.tol)):
                self.final_centroids=centroids
                return
            else:
                # continue iterating and count that iteration
                self.max_iter = self.max_iter-1
        # if after all the iterations we still don't get an acceptable error return
        self.final_centroids=centroids
        return

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        distance=(cdist(mat, self.final_centroids,metric=self.metric)) ## we calcuate the distance of the matrix points to any of the clusters
        points_classification=[]
        
        ## we assign the points to their closer centroid 
        for i in distance:
            points_classification.append(np.argmin(i))
        points_classification=np.array(points_classification)
        return(points_classification)


    def _get_error(self,centroids,mat,points_classification) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        new_error=[]
        for idx in range(self.k):
            ## get the distance from all the points in the cluster to their centroids
            distance = (cdist(mat[points_classification==idx], centroids))
            distance.tolist()
            # Ge the distance  correspoding to the current cluster
            error_d = np.array([item[idx] for item in distance])
            ## Get the mean error dor the distance abd square it
            new_error.append(np.mean((error_d)**2))
        return(np.array(new_error))
    
    def _get_centroids(self,mat,points_classification) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        centroids=[]
        for i in range(self.k):
            ## update the new centroids to the center of the current cluster
            centroids_new = mat[points_classification == i].mean(axis=0)
            centroids.append(centroids_new)
        return centroids
