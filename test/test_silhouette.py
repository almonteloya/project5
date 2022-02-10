import pytest
import numpy as np
from cluster import (
		KMeans, 
		Silhouette, 
		make_clusters,
		plot_clusters,
		plot_multipanel)
from scipy.spatial.distance import cdist


def test_silhouette():

	""" 
	Check the siloutte score implemetation
	Test: mean silohete scores will be higher if scale is lower. And viceversa
	"""
	clusters, labels = make_clusters(n=500, k=3,scale=.2)
	km = KMeans(k=3)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	meansocrehigh = (np.mean(scores))


	clusters, labels = make_clusters(n=500, k=3,scale=5)
	km = KMeans(k=3)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	meansocrelow = (np.mean(scores))

	assert(meansocrehigh>meansocrelow)

	return()

test_silhouette()
