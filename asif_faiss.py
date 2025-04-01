import sklearn
from tqdm import tqdm
import pickle
import torch
import faiss

class ASIF:

    def __init__(self, embeddings1, embeddings2,  p=8, k=800):

        self.embeddings1 = ASIF.__retrieve_or_assign__(embeddings1)
        self.embeddings2 = ASIF.__retrieve_or_assign__(embeddings2)

        self.p = p
        self.k = k

        self.space1_index = faiss.IndexFlatL2(self.embeddings1.numpy())

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

def extract_candidate_sets_from_clusters(kmeans, items, retrieve_ids=False):
    candidates_text = {cluster_id : set() for cluster_id in range(kmeans.cluster_centers_.shape[0])}
    for idx, i in enumerate(kmeans.labels_):
        if not retrieve_ids:
            candidates_text[i].add(items[idx])
        else:
            candidates_text[i].add(idx)
    return {cluster_id : list(candidates_text[cluster_id]) for cluster_id in range(kmeans.cluster_centers_.shape[0])}