
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import pickle
from asif import ASIF, extract_candidate_sets_from_clusters
import sys
from operator import itemgetter
import pandas as pd


data_folder = "data"

print("Loading text encoder...")
model = SentenceTransformer('xlm-roberta-base')

print("Loading chord encorder")
chord_encoder = SentenceTransformer('jammai/chocolm-modernbert-base')

print("Loading dataset...")
lyrics = pickle.load(open(f"{data_folder}/lyrics.pkl", "rb"))
chords = pickle.load(open(f"{data_folder}/chords.pkl", "rb"))
artist_song = pickle.load(open(f"{data_folder}/artist_song.pkl", "rb"))

print("Loading precomputed embeddings...")
chords_embeddings = pickle.load(open(f"{data_folder}/chords_embeddings_sbert_chocolm.pkl", "rb"))
lyrics_embeddings = pickle.load(open(f"{data_folder}/lyrics_embeddings_sbert_roberta.pkl", "rb"))

kmeans_lyrics = pickle.load(open(f"{data_folder}/lyrics_kmeans.pkl", "rb"))
kmeans_chords = pickle.load(open(f"{data_folder}/chords_kmeans.pkl", "rb"))

lyrics_candidates = extract_candidate_sets_from_clusters(kmeans_lyrics, lyrics, retrieve_ids=True)
chords_candidates = extract_candidate_sets_from_clusters(kmeans_chords, chords, retrieve_ids=True)

print("Initialising asif...")
asif = ASIF(
    lyrics_candidates,
    chords_candidates,
    torch.from_numpy(kmeans_lyrics.cluster_centers_),
    torch.from_numpy(kmeans_chords.cluster_centers_),
    lyrics_embeddings,
    chords_embeddings
)

def pack_results(candidates_ids, group_by_chord_progression=False):
    resulting_chords = itemgetter(*candidates_ids)(chords)
    resulting_lyrics = map(lambda x: x.strip(), itemgetter(*candidates_ids)(lyrics))
    resulting_artist = map(lambda x : x["artist"], itemgetter(*candidates_ids)(artist_song))
    resulting_song = map(lambda x : x["song"], itemgetter(*candidates_ids)(artist_song))
    df = pd.DataFrame({"chords": resulting_chords, "lyrics": resulting_lyrics, "artist": resulting_artist, "song": resulting_song})
    df = df.drop_duplicates()
    df = df[~df["chords"].str.contains("N")]
    if(group_by_chord_progression):
        grouped = df.groupby(by="chords")
        return {k:df.loc[v][["lyrics", "artist", "song"]].values.tolist() for (k,v) in grouped.groups.items()}
    else:
        return df.values.tolist()

def get_best_chord_sequence_candidate(lyrics):
    text_embedding = torch.from_numpy(model.encode(lyrics))
    lyrics_relative_coordinates = asif.compute_relative_coordinates_vs_space1(text_embedding)
    pred_candidate = (1 / (1 + torch.cdist(lyrics_relative_coordinates, asif.candidate_embeddings2_rc))).argmax(dim=1)
    candidates_ids = itemgetter(*pred_candidate.tolist())(chords_candidates)
    return pack_results(candidates_ids, group_by_chord_progression=True)

def get_best_lyrics_candidate(chord_sequence):
    chords_embeddings = torch.from_numpy(chord_encoder.encode(chord_sequence))
    chords_relative_coordinates = asif.compute_relative_coordinates_vs_space2(chords_embeddings)
    pred_candidate = (1 / (1 + torch.cdist(chords_relative_coordinates, asif.candidate_embeddings1_rc))).argmax(dim=1)
    candidates_ids = itemgetter(*pred_candidate.tolist())(chords_candidates)
    return pack_results(candidates_ids)

#print(get_best_chord_sequence_candidate(sys.argv))