import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

class DBSCAN3D:
    def __init__(self, X, eps=0.25, min_pts=7):
        self.X = np.array(X)  # Data points
        self.eps = eps  # Epsilon for neighborhood radius
        self.min_pts = min_pts  # Minimum points for core points
        self.labels = [0] * len(X)  # 0: unvisited, -1: noise
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
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=[color], alpha=0.7)
            ax.set_xlim(-3, 3)
        ax.set_ylim(0, 3)
        ax.set_zlim(-2, 2)
         # Find the largest cluster
        cluster_sizes = self.get_cluster_sizes()
        largest_cluster_idx = np.argmax(cluster_sizes)
        largest_cluster_label = list(set(labels))[largest_cluster_idx]
        print(largest_cluster_idx,largest_cluster_label)
        # Get points in the largest cluster
        largest_cluster_points = self.X[np.array(labels) == largest_cluster_label]
        
        # Compute the bounding box of the largest cluster
        x_min, y_min, z_min = np.min(largest_cluster_points, axis=0)
        x_max, y_max, z_max = np.max(largest_cluster_points, axis=0)

        # Create the bounding box corners
        bbox_corners = [
            [x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_min, y_max, z_max],
            [x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_min], [x_max, y_max, z_max]
        ]

        # Plot the six sides of the bounding box
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom rectangle
            [4, 5], [5, 7], [7, 6], [6, 4],  # Top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines connecting top and bottom
        ]
        
        for edge in edges:
            start, end = edge
            ax.plot([bbox_corners[start][0], bbox_corners[end][0]],
                    [bbox_corners[start][1], bbox_corners[end][1]],
                    [bbox_corners[start][2], bbox_corners[end][2]], color='red')
        ax.set_title("DBSCAN Clustering Results")
        plt.pause(0.1)
        save_path = 'dbscan_clusters.png'
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved as {save_path}")
        plt.show()
        


mat_file = scipy.io.loadmat('C:\\Users\\sica\\Desktop\\all_points.mat')
sorted(mat_file.keys())
X = mat_file['all_points']
mat_file = scipy.io.loadmat('C:\\Users\\sica\\Desktop\\10meas.mat')
sorted(mat_file.keys())
Y = mat_file['all_points']
combined = np.vstack((X, Y))


    # Initialize DBSCAN with example parameters
dbscan = DBSCAN3D(combined, eps=0.25, min_pts=7)

    # Get the number of points in each cluster
cluster_sizes = dbscan.get_cluster_sizes()
print("Number of points in each cluster:", cluster_sizes)

    # Get the average distance between points in each cluster
avg_distances = dbscan.get_average_distance_in_cluster()
print("Average distances between points in each cluster:", avg_distances)
numelements=len(cluster_sizes)
print(numelements)
dbscan.plot_clusters()

