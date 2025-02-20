import pickle
import torch
import sklearn
from tqdm import tqdm
import numpy as np

text_embeddings = torch.vstack(pickle.load(open("text_embeddings_ls.pkl", "rb")))

kmeans_text = sklearn.cluster.MiniBatchKMeans(n_clusters=300).fit(text_embeddings.numpy())
pickle.dump(kmeans_text, open("kmeans_text.pkl", "wb"))

text_sim = (1 / (1 + sklearn.metrics.pairwise_distances(kmeans_text.cluster_centers_, text_embeddings, metric="l2")))

k = 800
p = 8

relative_coordinates_text_candidate = []

for s in tqdm(text_sim):
    first_k_indexes = np.flip(np.argsort(s))[:k]
    relative_coordinates_text_candidate.append(np.array([p if idx in first_k_indexes else 0 for idx in range(len(s))]))

pickle.dump(relative_coordinates_text_candidate, open("relative_coordinates_text_candidate.pkl", "wb"))