import pickle
import torch
import sklearn
from tqdm import tqdm
import numpy as np
from asif import ASIF, extract_candidate_sets_from_clusters, compute_embedding
from transformers import AutoTokenizer, AutoModel
from matplotlib import pyplot as plt
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


embedding_mode = "hs"

train_lyrics = pickle.load(open("experimental_data/train_lyrics.pkl", "rb"))
train_chords = pickle.load(open("experimental_data/train_chords.pkl", "rb"))

test_lyrics = pickle.load(open("experimental_data/test_lyrics.pkl", "rb"))
test_chords = pickle.load(open("experimental_data/test_chords.pkl", "rb"))

train_lyrics_embeddings = pickle.load(open(f"experimental_data/train_lyrics_embeddings_{embedding_mode}.pkl", "rb"))
train_chords_embeddings = pickle.load(open(f"experimental_data/train_chords_embeddings_{embedding_mode}.pkl", "rb"))

test_lyrics_embeddings = pickle.load(open(f"experimental_data/test_lyrics_embeddings_{embedding_mode}.pkl", "rb"))
test_chords_embeddings = pickle.load(open(f"experimental_data/test_chords_embeddings_{embedding_mode}.pkl", "rb"))

# choose k = 8 for lyrics
kmeans_lyrics = sklearn.cluster.MiniBatchKMeans(n_clusters=8)
pickle.dump(kmeans_lyrics, open(f"experimental_data/kmeans_lyrics_{embedding_mode}.pkl", "wb"))

# choose k = 16 for chords
kmeans_chords = sklearn.cluster.MiniBatchKMeans(n_clusters=16)
pickle.dump(kmeans_chords, open(f"experimental_data/kmeans_chords_{embedding_mode}.pkl", "wb"))

lyrics_candidates = extract_candidate_sets_from_clusters(kmeans_lyrics, train_lyrics)
chords_candidates = extract_candidate_sets_from_clusters(kmeans_chords, train_chords)

asif = ASIF(
    lyrics_candidates,
    chords_candidates,
    torch.from_numpy(kmeans_lyrics.cluster_centers_),
    torch.from_numpy(kmeans_chords.cluster_centers_),
    train_lyrics_embeddings,
    train_chords_embeddings
)

chunk_size = 1000
predictions = []

for i in tqdm(range(0, len(test_lyrics_embeddings), chunk_size)):
    test_relative_coordinates_lyrics = asif.compute_relative_coordinates_vs_space1(test_lyrics_embeddings[i:i+chunk_size])
    test_relative_coordinates_lyrics = test_relative_coordinates_lyrics.to
    pred_chunk = (1 / (1 + torch.cdist(test_relative_coordinates_lyrics, asif.candidate_embeddings2_rc))).argmax(dim=1)
    predictions.extend(pred_chunk)

pickle.dump(predictions, open(f"experimental_data/predictions_{embedding_mode}.pkl", "wb"))