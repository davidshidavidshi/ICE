import numpy as np
from scipy.stats import entropy


def update_count_matrix(count_matrix, state, total_count, search_idx):
    idx = np.round(state*255).astype(int)
    for i in range(len(idx)):
        for j in range(len(idx[0])):
            try:
                place_idx = search_idx[idx[i,j]]
            except:
                search_idx[idx[i,j]] = len(search_idx)
                count_matrix = np.concatenate((count_matrix, np.zeros((40,40,1))), axis=2)
                place_idx = search_idx[idx[i,j]]
            count_matrix[i,j,place_idx] += 1
    total_count += 1
    return count_matrix, total_count, search_idx

def calculate_entropy(count_matrix, total_count):
    prob_matrix = count_matrix/total_count
    entropy_matrix = entropy(prob_matrix,axis=2,base=2)
    total_entropy = np.sum(entropy_matrix)
    return total_entropy

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def reset_intrinsic_param():
    total_count = 0
    last_entropy = 0
    count_matrix = np.zeros((40,40,1))
    search_idx = {0:0}
    return total_count, last_entropy, count_matrix, search_idx
