## Steps to do K-means clustering (EM):
# 1. Initialize K centers (u1, u2, ...uk) randomly
# 2. Calculate distance of each point to K centers
# 3. Assign a center to each point
# 4. Update k centers by taking the mean of all points assigned to each center
# 5. Repeat 2-4 until converge (k centers not moving)

## Implementation:
# 1. Initialize K centers (u1, u2, ...uk) randomly
# 2. Repeat until convergence: {
#    for every i, set ci = argmin(j)(xi - uj)^2
#    for every j, set uj = mean(all xi with ci = j)
# }

import random
import math

class KMeans:
    def __init__(self):
        pass

    def main(self, data, k):
        """
        data = [(x_0, y_0), (x_1, y_1), ... (x_n, y_n)]
        k = 3
        """
        centroids = self.initialize_centroids(data, k) # Time complexity: O(N); Space compexity: O(k)
        while True:
            centroids_old = centroids
            labels = self.get_labels(data, centroids) # Time complexity: O(N*k); Space complexity: O(N)
            centroids = self.update_centroids(data, labels, k) # Time complexity: O(N+k)

            if self.should_stop(centroids_old, centroids):
                break
        return labels

    def initialize_centroids(self, data, k):
        # 1. Initialize K clusters randomly from data.
        # return random.sample(data, k)
        # 2. Initialize K clusters randomly from a range of (min, max) from data.
        all_xs, all_ys = zip(*data) # Space complexity: O(N)
        x_min = min(all_xs) # Time complexity: O(n)
        y_min = min(all_ys)
        x_max = max(all_xs)
        y_max = max(all_ys)

        centroids = []
        for i in range(k):  # Time complexity: O(k)
            centroids.append([self.random_sample(x_min, x_max), self.random_sample(y_min, y_max)])
        return centroids
    
    def random_sample(self, low, high):
        return low + (high-low) * random.random()

    def get_labels(self, data, centroids):
        labels = []
        for point in data: # Time complexity: O(N)
            min_dist = float('inf')
            label = None
            for i, centroid in enumerate(centroids): # Time complexity: O(k)
                new_dist = self.get_distance(point, centroid)
                if new_dist < min_dist:
                    min_dist = new_dist
                    label = i
            labels.append(label)
        return labels

    def get_distance(self, point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def update_centroids(self, data, labels, k):
        new_centroids = [[0, 0] for i in range(k)]
        counts = [0] * k

        for point, label in zip(data, labels): # Time complexity: O(N)
            new_centroids[label][0] += point[0]
            new_centroids[label][1] += point[1]
            counts[label] += 1
        
        for i, (x, y) in enumerate(new_centroids): # Time complexity: O(k)
            new_centroids[i] = (x/counts[i], y/counts[i]) if counts[i] > 0 else (0, 0)
        return new_centroids

    def should_stop(self, centroids_old, centroids_new, threshold=1e-5):
        total_movement = 0
        for c_old, c_new in zip(centroids_old, centroids_new): # Time complexity: O(k)
            movement = self.get_distance(c_old, c_new)
            total_movement += movement
        if total_movement < threshold:
            return True
        return False



data = [[0, 0], [0, 1], [0, 2], [1, 1], [2, 2], [0, 0.9], [0, 1.9], [1.1, 1.1], [2.1, 2.1]]
k = 3

kmeans = KMeans()
centroids = kmeans.main(data, k)
print(centroids)