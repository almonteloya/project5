# Write your k-means unit tests here
import pytest
import numpy as np
from cluster import (
		KMeans, 
		Silhouette, 
		make_clusters,
		plot_clusters,
		plot_multipanel)
from scipy.spatial.distance import cdist


def test_kmeans():

	""" 
	Check K means algorithm 
	Test: raise error when k=0
	"""
	try:
		kmeans = KMeans(k=0)
		assert False
	except ValueError:
		assert True

	"""
	Test: when k=1 all the label predicted should be only one group
	"""

	clusters, labels = make_clusters(n=50, k=1,scale=1)
	km = KMeans(k=1)
	km.fit(clusters)
	pred = km.predict(clusters)
	## all the pred are the same 
	assert(len(set(pred))== 1)


	"""
	Tight clustering. 

	Test: Generate two clusters with small scale. 
	Because we set the seed we know the exact number of points per cluster. 
	Check that the proportion of the predicted label is the same as the generated label.
	"""
	clusters, labels = make_clusters(n=100, k=2,scale=.1)
	km = KMeans(k=2)
	km.fit(clusters)
	pred = km.predict(clusters)

    ## convert to list so we can count
	l_pred=pred.tolist()
	labels_l=labels.tolist()
    
	a={l_pred.count(0),l_pred.count(1)}
	b={labels_l.count(0),labels_l.count(1)}
    
    ## There should be no differences between the two sets 
	assert(len(a.difference(b)) == 0)


	"""
	Large clustering 
	Test: Use a large K, make sure we have the correct cluster number

	"""

	clusters, labels = make_clusters(k=70)
	km = KMeans(k=70)
	km.fit(clusters)
	pred = km.predict(clusters)
    
	assert (len(set(pred)) == 70)

test_kmeans()

