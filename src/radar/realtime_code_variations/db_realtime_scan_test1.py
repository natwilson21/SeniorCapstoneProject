import sys
sys.path.append('C:/Program Files/MATLAB/R2019b/extern/engines/python')
import matlab.engine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import numpy as np  # Using NumPy for easier matrix manipulation
import scipy.io  # Import the scipy.io module for saving .mat files

class DBSCAN3D:
    def __init__(self, X, eps=0.25, min_pts=7):
        # Make sure X is a 2D array of shape (N, 3)
        self.X = np.vstack(X) if isinstance(X, list) else X  # Stack arrays if X is a list of arrays
        self.eps = eps  # Epsilon for neighborhood radius
        self.min_pts = min_pts  # Minimum points for core points
        self.labels = [0] * len(self.X)  # Initialize labels with the correct length based on X
        self.cluster_id = 0  # Cluster ID counter
    
    def plot_data_3d(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha=0.7)
        ax.set_title("Sample Data for DBSCAN")
        plt.show()
    
    def plot_spheres(self, X, eps, ax):
        for point in X:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = eps * np.outer(np.cos(u), np.sin(v)) + point[0]
            y = eps * np.outer(np.sin(u), np.sin(v)) + point[1]
            z = eps * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]
            ax.plot_surface(x, y, z, color='b', alpha=0.1, rstride=4, cstride=4)
    
    def plot_epsilon_neighborhoods(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha=0.7)
        self.plot_spheres(self.X[:5], self.eps, ax)
        ax.set_title(f"Epsilon Neighborhoods (eps={self.eps})")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], alpha=0.7)
        ax2.scatter(self.X[0, 0], self.X[0, 1], self.X[0, 2], s=100, c='red')
        ax2.set_title(f"Core Point (min_pts={self.min_pts})")
        plt.show()
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def get_neighbors(self, point_idx):
        distances = [self.euclidean_distance(self.X[point_idx], other_point) for other_point in self.X]
        return [i for i, dist in enumerate(distances) if dist <= self.eps]
    
    def find_core_points(self):
        core_points = []
        for i in range(len(self.X)):
            if len(self.get_neighbors(i)) >= self.min_pts:
                core_points.append(i)
        return core_points
    
    def expand_cluster(self, labels, point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -1:  # Noise becomes border point
                labels[neighbor] = cluster_id
            elif labels[neighbor] == 0:  # Unvisited
                labels[neighbor] = cluster_id
                new_neighbors = self.get_neighbors(neighbor)
                if len(new_neighbors) >= self.min_pts:
                    neighbors.extend(new_neighbors)
            i += 1
        return labels
    
    def dbscan(self):
        labels = self.labels.copy()
        core_points = self.find_core_points()
        
        for point_idx in range(len(self.X)):
            if labels[point_idx] != 0:
                continue
            if point_idx in core_points:
                self.cluster_id += 1
                neighbors = self.get_neighbors(point_idx)
                labels = self.expand_cluster(labels, point_idx, neighbors, self.cluster_id)
            else:
                labels[point_idx] = -1  # Noise
        
        return labels
    
    def get_cluster_sizes(self):
        labels = self.dbscan()
        cluster_sizes = []
        for label in set(labels):
            if label != -1:  # Ignore noise
                cluster_sizes.append(labels.count(label))  # Count the number of points in this cluster
        return cluster_sizes
    
    def get_average_distance_in_cluster(self):
        labels = self.dbscan()
        cluster_avg_distances = []

        # For each cluster, calculate the average distance between all pairs of points
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Ignore noise
                continue
            cluster_points = self.X[np.array(labels) == label]
            total_distance = 0
            count = 0
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    total_distance += self.euclidean_distance(cluster_points[i], cluster_points[j])
                    count += 1
            if count > 0:
                cluster_avg_distances.append(total_distance / count)
            else:
                cluster_avg_distances.append(0)  # If no pairs, append 0
        
        return cluster_avg_distances
    
    def plot_clusters(self):
        labels = self.dbscan()
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'gray'  # Use gray for noise points
            class_member_mask = (np.array(labels) == label)
            xy = self.X[class_member_mask]
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=[color], alpha=0.7, label=f'Cluster {label}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 5)
        ax.set_zlim(-5, 5)
        ax.set_title("DBSCAN Clustering Results")
        ax.legend()
        plt.pause(0.1)

# Initialize the loop counter
i = 1

# Initialize an empty list to hold all points
all_points = []

# While loop to continuously retrieve points and append them
while True:
    eng = matlab.engine.start_matlab()
    eng.run('C:/Users/sica/Downloads/mm-wave-master/mm-wave-master/client_center.m', nargout=0)
    # Retrieve the variable 'A' from MATLAB
    points = eng.workspace['A']

    # Convert MATLAB array to a Python list or NumPy array (depending on the type)
    points = np.array(points)  # Convert to a NumPy array (adjust if you need a list)

    # Append the points to the all_points list
    all_points.append(points)
    dbscan = DBSCAN3D(all_points, eps=0.25, min_pts=7)

    # Get the number of points in each cluster
    cluster_sizes = dbscan.get_cluster_sizes()
    print("Number of points in each cluster:", cluster_sizes)

    # Get the average distance between points in each cluster
    avg_distances = dbscan.get_average_distance_in_cluster()
    print("Average distances between points in each cluster:", avg_distances)
    numelements = len(cluster_sizes)
    print(numelements)
    dbscan.plot_clusters()

    # Optionally, print the saved points (for debugging purposes)
    print(f"Iteration {i}: {points}")

    # Increment the loop counter
    i += 1
    eng.quit()

    # Wait for a short time before the next iteration
    time.sleep(1)
    print("2sleep")

    # You can add a condition to break the loop after a certain number of iterations
    if i > 10:  # Adjust this as needed
        break

plt.show()

# Convert the list of points to a single NumPy matrix
all_points_matrix = np.vstack(all_points)  # Stack arrays vertically to form a single matrix

# Save the matrix as a MATLAB .mat file
scipy.io.savemat('all_points.mat', {'all_points': all_points_matrix})
