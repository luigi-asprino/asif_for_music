from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from tqdm import tqdm
import gc
from torch.utils.data import DataLoader

def compute_embedding(tokenizer, model, input):
    tokenized = tokenizer(input, return_tensors="pt", padding=True)
    tokenized = tokenized.to(device)
    model_output = model(**tokenized, output_hidden_states=True)
    
    embedding = model_output.last_hidden_state[:, 0, :]
    embedding = embedding.to("cpu")

    hidden_states = []
    for n_input in range(len(input)):
        input_hd = []
        for n_layer in range(len(model_output.hidden_states)):
            input_hd.append(model_output.hidden_states[n_layer][n_input: n_input+1, :, :])
        hidden_states.append(torch.mean(torch.vstack(input_hd), dim=(0,1)))
    hidden_states = torch.vstack(hidden_states)
    embedding_hs = hidden_states.to("cpu")

    return embedding, embedding_hs

device = torch.device('cuda')

chord_tokenizer = AutoTokenizer.from_pretrained("jammai/chocolm-modernbert-base-transposed")
chord_model = AutoModel.from_pretrained("jammai/chocolm-modernbert-base-transposed")
chord_model.to(device)

text_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
text_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
text_model.to(device)

chords2lyrics = pickle.load(open("harte_shongs.pkl", "rb"))

chords_lyrics = []

print("extract verse chords and lyrics")

for song in tqdm(chords2lyrics):
    lyrics = eval(song["lyrics"])
    for verse in song["verse_to_harte_chords"]:
        if verse + 1 in lyrics and len(lyrics[verse + 1].rstrip()):
            chords_lyrics.append({"chords": " ".join(song["verse_to_harte_chords"][verse]), "lyrics": lyrics[verse + 1]})

pickle.dump(chords_lyrics,open(f"chords_lyrics.pkl", "wb"))

chord_embeddings_hs = []
text_embeddings_hs = []

chord_embeddings_ls = []
text_embeddings_ls = []

data_loader = DataLoader(chords_lyrics, batch_size=256)

with torch.no_grad():
    for batch in tqdm(data_loader):
        
        # compute text embedding
        text_embedding_ls, text_embedding_hs = compute_embedding(text_tokenizer, text_model, batch["lyrics"])
        text_embeddings_ls.append(text_embedding_ls)
        text_embeddings_hs.append(text_embedding_hs)

        # compute chord embedding
        chord_embedding_ls, chord_embedding_hs = compute_embedding(chord_tokenizer, chord_model, batch["chords"])
        chord_embeddings_ls.append(chord_embedding_ls)
        chord_embeddings_hs.append(chord_embedding_hs)


pickle.dump(chord_embeddings_ls, open(f"chord_embeddings_ls.pkl", "wb"))
pickle.dump(text_embeddings_ls, open(f"text_embeddings_ls.pkl", "wb"))

pickle.dump(chord_embeddings_hs, open(f"chord_embeddings_hs.pkl", "wb"))
pickle.dump(text_embeddings_hs, open(f"text_embeddings_hs.pkl", "wb"))

