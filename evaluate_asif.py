import pickle
import torch
import sklearn
from tqdm import tqdm
import numpy as np
from asif import ASIF, extract_candidate_sets_from_clusters
from transformers import AutoTokenizer, AutoModel
from matplotlib import pyplot as plt
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

train_lyrics = pickle.load(open("experimental_data/train_lyrics.pkl", "rb"))
train_chords = pickle.load(open("experimental_data/train_chords.pkl", "rb"))

test_lyrics = pickle.load(open("experimental_data/test_lyrics.pkl", "rb"))
test_chords = pickle.load(open("experimental_data/test_chords.pkl", "rb"))

embedding_mode_lyrics = "roberta_sbert_embeddings"
embedding_mode_chords = "chocolm_sbert_embeddings"

train_lyrics_embeddings = pickle.load(open(f"experimental_data/embeddings/train_lyrics_{embedding_mode_lyrics}.pkl", "rb"))
train_chords_embeddings = pickle.load(open(f"experimental_data/embeddings/train_chords_{embedding_mode_chords}.pkl", "rb"))

test_lyrics_embeddings = pickle.load(open(f"experimental_data/embeddings/test_lyrics_{embedding_mode_lyrics}.pkl", "rb"))
test_chords_embeddings = pickle.load(open(f"experimental_data/embeddings/test_chords_{embedding_mode_chords}.pkl", "rb"))

kmeans_lyrics = pickle.load(open(f"experimental_data/best_kmeans_lyrics_{embedding_mode_lyrics}.pkl", "rb"))
kmeans_chords = pickle.load(open(f"experimental_data/best_kmeans_chords_{embedding_mode_chords}.pkl", "rb"))

#chords_embeddings = torch.vstack([train_chords_embeddings, test_chords_embeddings])
#chords_sequences = train_chords + test_chords

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
chunk_size = 500
predictions = []

for i in tqdm(range(0, len(test_lyrics_embeddings), chunk_size)):
    test_relative_coordinates_lyrics = asif.compute_relative_coordinates_vs_space1(test_lyrics_embeddings[i:i+chunk_size])
    pred_chunk = (1 / (1 + torch.cdist(test_relative_coordinates_lyrics, asif.candidate_embeddings2_rc))).argmax(dim=1)
    predictions.extend(pred_chunk)

pickle.dump(predictions, open(f"experimental_data/predictions_{embedding_mode_lyrics}_{embedding_mode_chords}.pkl", "wb"))