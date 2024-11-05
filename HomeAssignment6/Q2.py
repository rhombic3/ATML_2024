import numpy as np
import matplotlib.pyplot as plt

# set parameters
T = 100
n_clients = 4
theta_star = np.array([0, 0])
alpha = 0.5
lamb = 2
zeta_o = 0.1
rho_o = 0.1

# initial theta
theta_i_values = np.random.rand(n_clients, 2) * 10

# calculate the error with different algebraic connectivity
def compute_algebraic_connectivity_error(mu_values):
    errors = []
    for mu in mu_values:
        error = []
        for t in range(1, T + 1):
            rho = 1 - alpha*mu
            error_t = (zeta_o**2 / t**(2/3)) + (rho**2 * lamb / t)
            error.append(error_t)
        errors.append(error)
    return errors

# calculate the error with different gradient dissimilarity
def compute_gradient_dissimilarity_error(zeta_values):
    errors = []
    for zeta in zeta_values:
        error = []
        for t in range(1, T + 1):
            error_t = (zeta**2 / t**(2/3)) + (rho_o**2 * lamb / t)
            error.append(error_t)
        errors.append(error)
    return errors

# set the range of algebraic connectivity and gradient dissimilarity
mu_values = np.linspace(0.1, 1.0, 5)
zeta_values = np.linspace(0.1, 1.0, 5)

# calculate error
errors_mu = compute_algebraic_connectivity_error(mu_values)
errors_zeta = compute_gradient_dissimilarity_error(zeta_values)

# plot
plt.figure(figsize=(12, 6))
for i, mu in enumerate(mu_values):
    plt.plot(range(1, T + 1), errors_mu[i], label=f'uG={mu:.2f}')
plt.xlabel('Iterations (t)')
plt.ylabel('Error')
plt.title('(a) Impact of Algebraic Connectivity on Error')
plt.legend()
plt.grid(True)
plt.savefig("a_result.png")

plt.figure(figsize=(12, 6))
for i, zeta in enumerate(zeta_values):
    plt.plot(range(1, T + 1), errors_zeta[i], label=f'ζ={zeta:.2f}')
plt.xlabel('Iterations (t)')
plt.ylabel('Error')
plt.title('(b) Impact of Gradient Dissimilarity on Error')
plt.legend()
plt.grid(True)
plt.savefig("b_result.png")
