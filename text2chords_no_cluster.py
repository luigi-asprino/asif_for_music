
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import pickle
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


def get_best_chord_sequence_candidate(lyrics, threshold=0):
    text_embedding = torch.from_numpy(model.encode(lyrics))
    lyrics_dist = torch.cdist(text_embedding, lyrics_embeddings)
    min_dist = torch.unique(lyrics_dist.sort().values)[threshold]
    indexes = (lyrics_dist <= min_dist).nonzero(as_tuple=True)[1]
    return pack_results(indexes, group_by_chord_progression=True)

def get_best_lyrics_candidate(chord_sequence, threshold=0):
    chords_embedding = torch.from_numpy(chord_encoder.encode(chord_sequence))
    chords_dist = torch.cdist(chords_embedding, chords_embeddings)
    min_dist = torch.unique(chords_dist.sort().values)[threshold]
    candidates_ids = (chords_dist == min_dist).nonzero(as_tuple=True)[1]
    return pack_results(candidates_ids)

#print(get_best_chord_sequence_candidate(sys.argv))