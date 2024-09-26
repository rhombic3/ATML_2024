import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# !curl -s https://rasmuspagh.net/data/glove.twitter.27B.100d.names.pickle -O

def compute_cost(points, centers):
    distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
    return np.mean(np.min(distances_squared, axis=0))

# Generate gaussian noise
def gaussian_noise(size, sigma):
    return np.random.normal(0, sigma, size)

def k_means(points, k, t, n, rho):
    # Calculate sigma and sigma' based on the privacy loss rho
    sigma = np.sqrt(4 * t / rho)
    sigma_prime = np.sqrt(2 * t / rho)

    initial_assignment = np.random.choice(range(k), n) # Randomly select n points from [0, ..., k-1]
    cluster_indexes = [ (initial_assignment == i) for i in range(k) ] # Grouping
    cluster_sizes = [ cluster_indexes[i].sum() for i in range(k) ] # Get the size g of each group

    for l in range(t):
        cluster_sums = [ np.sum(points[cluster_indexes[i]], axis=0) + gaussian_noise(points.shape[1], sigma) for i in range(k) ] # Get the sum of each set of vectors f
        centers = np.array([ cluster_sums[i] / max(1, cluster_sizes[i]) for i in range(k) ]) # Get the cluster center c
        
        distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
        assignment = np.argmin(distances_squared, axis=0) # Find the closest cluster center
        cluster_indexes = [ (assignment == i) for i in range(k) ]
        cluster_sizes = [cluster_indexes[i].sum() + gaussian_noise(1, sigma_prime) for i in range(k)]

    return centers

if __name__ == "__main__":
    k = 5 # Number of clusters
    t_range = range(1,10)
    costs = []

    # Privacy loss rho
    rho = 0.001

    # Read the input points and shape
    input_file = "glove.twitter.27B.100d.names.pickle"
    with open(input_file, 'rb') as f:
        embedding = pickle.load(f)
    names = list(embedding.keys())
    points = np.array([ embedding[x] for x in names ])
    n, d = points.shape

    for t in t_range: # number of iterations
        centers = k_means(points, k, t, n, rho)
        costs.append(compute_cost(points, centers))

    fig, ax = plt.subplots()
    ax.set_xlabel('t')
    ax.set_ylabel('cost with privacy')
    ax.plot(t_range, costs)
    plt.xscale('log')

    if not os.path.exists("result"):
        os.makedirs("result")

    plt.savefig("result/cost_privacy.png")