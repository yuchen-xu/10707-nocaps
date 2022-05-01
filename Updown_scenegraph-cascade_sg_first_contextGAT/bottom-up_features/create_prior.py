import numpy as np

prob_matrix = np.zeros((151, 151, 51), dtype=np.int64)

np.save("freq_prior.npy", prob_matrix)