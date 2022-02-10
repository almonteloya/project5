import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        self.metric = metric ## we give the atribute

    def _mean_distance_cluster (self, matrix1 , matrix2):
       
        """
        
        Calculates the distance between points. Returns the average.

        inputs:

        matrix1,matrix2: 2D np.arrays containing coordinates

        Output:

        n.p array containing the average distance per column


        """
        distance_cluster = cdist(matrix1,matrix2,metric=self.metric) ## we calculate the distance between each point in the cluster
        
        return(np.mean(distance_cluster,axis=1)) ## this is the average for each point whitin the same cluster
    
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if len(set(y))<=1:
            raise ValueError("Cannot do Silhouette with less than two clusters")


        labels_predicted = y
        mat =  X

        keys = set(labels_predicted) ## This keeps the unique clusters we are workign with
        silohete = []
        dicts={} 
        for i in keys:
            ## Here we have as keys the cluster and as values all the points that correspond to each specific cluster
            dicts[i] = mat[labels_predicted == i]
        
        ## We iterate over each cluster, from 0 to k.

        for i in dicts:
            distance_intra=[]
            ai = self._mean_distance_cluster(dicts[i],dicts[i]) ## we calculate the average dstance whitin the cluster
            curr_cluster = list(keys)
            ## We calculate distance with points NOT in the cluster
            curr_cluster.remove(i)
            for y in curr_cluster:
                distance_intra.append(self._mean_distance_cluster(dicts[i],dicts[y]))
            bi_l=[]
            ## We look for the minimum distance to the other cluster
            for j in range(len(dicts[i])):
                bi_l.append(min([item[j] for item in distance_intra]))
            bi = np.array(bi_l) 

            ## Calculate silohette score using the formula b-a/max(a,b)   
            for x in range(len(ai)):
                val = bi[x]-ai[x]
                maxb = max(bi[x],ai[x])
                s_score = val/maxb
                silohete.append(s_score)
        
        # We now have a list of silohete scores, but they are ordered from 0 to k
        # This might not be the original order from labels_predicted so we need to reorder them

        start=0
        rs =list(labels_predicted)

        for k in keys:
            nl = len(labels_predicted[labels_predicted==k])
            nl=start+nl
            # The values in silohete are ordered by cluster
            selected = silohete[start:nl]
            start=nl
            # we order them the same way labels_predicted is ordered.
            for i in range(len(labels_predicted)):
                if labels_predicted[i]==k:
                    rs[i]=(selected.pop(0))
        # Don't forget to return the numpy array
        return(np.array(rs))