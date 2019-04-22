import unittest
import numpy as np
import random
import functions
from sklearn.cluster import KMeans

class TestPSO (unittest.TestCase):
    def test_davies_bouldin(self):
        data = np.array([[4239.49, 4275.38, 4290.26, 4326.67, 4745.13],
                [4230.77, 4269.23, 4284.62, 4328.72, 4738.97],
                [4243.59, 4271.79, 4287.18, 4329.74, 4735.90]])
        kmeans = KMeans(n_clusters=2).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        centers = centers.reshape(centers.shape[0]*centers.shape[1])
        DB = functions.davies_bouldin(data, labels)(centers)
        self.assertAlmostEqual(DB, 0.4654235639196072)


    def test_xie_beni(self):
        data = np.array([[4239.49, 4275.38, 4290.26, 4326.67, 4745.13],
                [4230.77, 4269.23, 4284.62, 4328.72, 4738.97],
                [4243.59, 4271.79, 4287.18, 4329.74, 4735.90]])
        kmeans = KMeans(n_clusters=2).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        centers = centers.reshape(centers.shape[0]*centers.shape[1])
        XB = functions.xie_beni(data, labels)(centers)
        self.assertAlmostEqual

if __name__ == '__main__':
    unittest.main()
