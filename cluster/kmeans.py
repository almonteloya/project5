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
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        ## pick a random number to initialize the clusters
        ran = np.random.choice(len(mat), self.k, replace=False,)
        ## This are the first clusters
        centroids = mat[ran, :]
        error = np.ones(self.k)* np.inf * -1 ### initialize the errors to a large value
        while (self.max_iter >= 0):
            distance=(cdist(mat, centroids,metric=self.metric)) ## we calcuate the distance of the matrix points to any of the clusters
            points_classification=[]
            for i in distance:
                points_classification.append(np.argmin(i))
            points_classification=np.array(points_classification)
            centroids = self.get_centroids(mat,points_classification) ## We call the get centroids part
            ## here I call for the error
            prev_error = error 
            error = self.get_error(centroids,mat,points_classification)
            if (np.all((error - prev_error)< self.tol)):
                self.final_centroids=centroids
                return
            else:
                self.max_iter = self.max_iter-1
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
        for i in distance:
            points_classification.append(np.argmin(i))
        points_classification=np.array(points_classification)
        return(points_classification)


    def get_error(self,centroids,mat,points_classification) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        new_error=[]
        for idx in range(self.k):
            distance = (cdist(mat[points_classification==idx], centroids))
            distance.tolist()
            error_d = np.array([item[idx] for item in distance])
            new_error.append(np.mean((error_d)**2))
        return(np.array(new_error))
    
    def get_centroids(self,mat,points_classification) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        centroids=[]
        for i in range(self.k):
            centroids_new = mat[points_classification==i].mean(axis=0)
            centroids.append(centroids_new)
        return centroids
