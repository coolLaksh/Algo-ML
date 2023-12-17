import numpy as np

def closest_Centroid(centroids, dataPoint):
    min_distance = float('inf')
    ret_idx = 0

    for idx, centroid in enumerate(centroids):
        dist = np.linalg.norm(centroid - dataPoint)
        if dist < min_distance:
            min_distance = dist
            ret_idx = idx

    return ret_idx

def K_Means(data, K, max_iterations=100):
    centroids = np.array(data[np.random.choice(data.shape[0], K, replace=True)])
    
    for iteration in range(max_iterations):
        cluster_points = {}
        
        for i in range(K):
            cluster_points[i] = []

        for dataPoint in data:
            cluster_K = closest_Centroid(centroids, dataPoint)
            cluster_points[cluster_K].append(dataPoint)

        new_centroids = []
        for k in range(K):
            new_centroids.append(np.mean(cluster_points[k], axis=0))

        new_centroids = np.array(new_centroids)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, cluster_points
