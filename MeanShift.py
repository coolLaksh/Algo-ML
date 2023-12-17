import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Mean_Shift:
    def __init__(self, bandwidth):
        self.Bandwidth = bandwidth
        self.centroids = {}

    def fit(self, data):
        centroids = {}

        # initalize all points as centroids
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            # for ith centroid calculate its data points within bandwidth
            for i in centroids:
                in_Bandwidth = []
                centroid = centroids[i]
                for feature in data:
                    if np.linalg.norm(feature - centroid) < self.Bandwidth:
                        in_Bandwidth.append(feature)

                new_centroid = np.average(in_Bandwidth, axis=0)
                new_centroids.append(tuple(new_centroids))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)
            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids
