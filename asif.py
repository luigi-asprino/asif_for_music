import sklearn
from tqdm import tqdm
import pickle
import torch

class ASIF:

    def __init__(self, candidates1, candidates2, candidate_embeddings1, candidate_embeddings2, embeddings1, embeddings2,  distance_function = "l2", p=8, k=800, candidate_embeddings1_rc = None, candidate_embeddings2_rc = None):
        
        self.candidates1 = ASIF.__retrieve_or_assign__(candidates1)
        self.candidates2 = ASIF.__retrieve_or_assign__(candidates2)

        self.embeddings1 = ASIF.__retrieve_or_assign__(embeddings1)
        self.embeddings2 = ASIF.__retrieve_or_assign__(embeddings2)

        self.candidate_embeddings1 = ASIF.__retrieve_or_assign__(candidate_embeddings1)
        self.candidate_embeddings2 = ASIF.__retrieve_or_assign__(candidate_embeddings2)

        self.distance_function = distance_function

        self.p = p
        self.k = k

        if candidate_embeddings1_rc == None:
            self.candidate_embeddings1_rc = ASIF.compute_relative_coordinates(self.candidate_embeddings1, self.embeddings1)
        else:
            self.candidate_embeddings1_rc = ASIF.__retrieve_or_assign__(candidate_embeddings1_rc)

        if candidate_embeddings2_rc == None:
            self.candidate_embeddings2_rc = ASIF.compute_relative_coordinates(self.candidate_embeddings2, self.embeddings2)
        else:
            self.candidate_embeddings2_rc = ASIF.__retrieve_or_assign__(candidate_embeddings2_rc)

    
    @staticmethod
    def __retrieve_or_assign__(e):
        if isinstance(e, str):
            return pickle.load(open(e, "rb"))
        else:
            return e


    @staticmethod
    def compute_relative_coordinates(candidate_embeddings, embeddings, k=800, p=8):
        
        sim = (1 / (1 + torch.cdist(candidate_embeddings, embeddings)))
        
        result = torch.zeros(sim.size())
        
        for i, j in enumerate(torch.argsort(sim, descending=True)[:,:k]):
            result[i][j] = p

        return torch.nn.functional.normalize(result)
    
    def compute_relative_coordinates_vs_space1(self, embeddings):
        return ASIF.compute_relative_coordinates(embeddings, self.embeddings1)
    
    def compute_relative_coordinates_vs_space2(self, embeddings):
        return ASIF.compute_relative_coordinates(embeddings, self.embeddings2)
    
    def most_similar_candidate_vs_space1(self, relative_embeddings):
        return (1 / (1 + torch.cdist(relative_embeddings, self.candidate_embeddings1_rc))).argmax()
    
    def most_similar_candidate_vs_space2(self, relative_embeddings):
        return (1 / (1 + torch.cdist(relative_embeddings, self.candidate_embeddings2_rc))).argmax()
    
def compute_embedding(tokenizer, model, input, device="cuda", mode="ls"):
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

    if mode == "ls":
        return embedding

    return embedding, embedding_hs

def extract_candidate_sets_from_clusters(kmeans, items):
    candidates_text = {cluster_id : set() for cluster_id in range(kmeans.cluster_centers_.shape[0])}
    for idx, i in enumerate(kmeans.labels_):
        candidates_text[i].add(items[idx])
    return {cluster_id : list(candidates_text[cluster_id]) for cluster_id in range(kmeans.cluster_centers_.shape[0])}