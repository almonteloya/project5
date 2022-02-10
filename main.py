import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

#def centeroidnp(arr):
#    length = arr.shape[0]
#    sum_x = np.sum(arr[:, 0])
#    sum_y = np.sum(arr[:, 1])
#    return sum_x/length, sum_y/length

def main():

    # create tight clusters
    clusters, labels = make_clusters(scale=0.3)
    plot_clusters(clusters, labels, filename="figures/tight_clusters.png")

    # create loose clusters
    clusters, labels = make_clusters(scale=2)
    plot_clusters(clusters, labels, filename="figures/loose_clusters.png")
    

    
#    clusters, labels = make_clusters(k=2,scale=3)
#    km = KMeans(k=2)
#    km.fit(clusters)
#    pred = km.predict(clusters)
#    scores = Silhouette().score(clusters, pred)
#    centroid = (centeroidnp(clusters[pred==0]))
#    np
#    l_pred=pred.tolist()
#    labels_l=labels.tolist()
#    plot_multipanel(clusters, labels, pred, scores,filename="figures/clusters_test.png")
    


    """
    uncomment this section once you are ready to visualize your kmeans + silhouette implementation
    """
    clusters, labels = make_clusters(k=4, scale=4)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    plot_multipanel(clusters, labels, pred, scores,filename="figures/clusters_main.png")
    

if __name__ == "__main__":
    main()
