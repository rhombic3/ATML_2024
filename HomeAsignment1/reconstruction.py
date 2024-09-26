import numpy as np
import requests as rq
from scipy.optimize import linprog

# Retrieve answer to challenge for a given query
def query(challenge_id, query_vector, submit=False):
    # Only alphanumeric challenge_id and vextor entries in {-1,+1} are allowed:
    assert(challenge_id.isalnum())
    assert(np.max(np.minimum(np.abs(query_vector-1),np.abs(query_vector+1)))==0)

    # if query array is 1d, make it 2d
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1,-1)

    payload = { 'challengeid': challenge_id, 'submit': submit,
                'query': str(query_vector.tolist()) }
    response = rq.post("https://baconbreaker.pythonanywhere.com/query", 
                       data = payload).json()
    if submit == False:
        return np.array(eval(response['result']))
    else:
        return response['result']

# Using reconstruction attack to get the hidden dataset
def reconstruction_attack(challenge_id, n, num_queries):
    # Conduct query
    queries = np.random.choice([-1,+1], size=(num_queries, n)) # Set of random queries
    query_results = query(challenge_id, queries)

    # According to the form Q - z <= sum(qx) <= Q + z, define the parameters
    c = np.concatenate((np.zeros(n), np.ones(num_queries)), axis=0)
    Aub = np.block([
        [queries, -np.eye(num_queries)],
        [-queries, -np.eye(num_queries)]
    ])
    bub = np.concatenate((query_results, -query_results), axis=0)

    # Upper and lower bounds for dataset entries and noise
    bounds = [(-1, 1)] * n + [(-100, 100)] * num_queries

    # Use the linear programming to minimize the noise
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method='highs')

    if res.success:
        # Return the best estimated dataset
        return np.sign(res.x[:n])
    else:
        # Return a random dataset
        print("Fail to linear programming, return a random dataset.")
        return(np.random.choice([-1,+1], size=n))
    
if __name__ == "__main__":
    challenge_id = 'rhombic3' # identifier for hidden dataset
    n = 100 # number of entries in hidden dataset
    num_queries = 2*n # number of queries to be asked

    reconstruction_dataset = reconstruction_attack(challenge_id, n, num_queries)

    # Calculate the correct probability
    best_query_result = query(challenge_id, reconstruction_dataset, submit=False)
    print(f"\nReconstruction attack achieves fraction {(1 + best_query_result / n) / 2} correct values")