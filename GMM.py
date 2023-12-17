from scipy.stats import norm
import numpy as np

def random_init(n_components, X):
    pi = np.ones((n_components))
    means = np.random.choice(X, n_components)
    variances = np.random.random_sample(size=n_components)
    
    return means, variances, pi

def step_expectation(X, n_components, means, variances):
    weights = np.zeros((n_components, len(X)))
    for j in range(n_components):
        weights[j, :] = norm(loc=means[j], scale=np.sqrt(variances[j])).pdf(X)
    return weights

def step_maximization(X, weights, means, variances, n_components, pi):
    responsibilities = []
    for j in range(n_components):  
        responsibilities.append((weights[j] * pi[j]) / (np.sum([weights[i] * pi[i] for i in range(n_components)], axis=0)))

        means[j] = np.sum(responsibilities[j] * X) / (np.sum(responsibilities[j]))
        variances[j] = np.sum(responsibilities[j] * np.square(X - means[j])) / (np.sum(responsibilities[j]))
    
        pi[j] = np.mean(responsibilities[j])

    return variances, means, pi

def train_gmm(data, n_components=3, n_steps=50):
    means, variances, pi = random_init(n_components, data)
    for step in range(n_steps):
        weights = step_expectation(data, n_components, means, variances)
        variances, means, pi = step_maximization(data, weights, means, variances, n_components, pi)
   
    return means, variances, pi
