import pickle
import faiss 
import torch
import numpy as np
import math
from asif import extract_candidate_sets_from_clusters
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import sklearn
from operator import itemgetter 
from typing import Tuple, List, Type, Union

def relative_represent(y: torch.Tensor, basis: torch.Tensor, non_zeros: int = 800, max_gpu_mem_gb: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the sparse decomposition of a tensor y with respect to a basis, 
    considering the available GPU memory.
    
    Args:
        y (torch.Tensor): Vectors to represent.
        basis (torch.Tensor): Basis to represent with respect to.
        non_zeros (int): Nonzero entries in the relative representation.
        max_gpu_mem_gb (int): Maximum GPU memory allowed to use in gigabytes.
        
    Returns:
        indices (torch.Tensor): Indices of the nonzero entries in each relative representation of y.
        values (torch.Tensor): Corresponding coefficients of the entries.
    """
    values, indices = torch.zeros((y.shape[0], non_zeros)), torch.zeros((y.shape[0], non_zeros), dtype=torch.long)

    free_gpu_mem = max_gpu_mem_gb * 1024 ** 3
    max_floats_in_mem = free_gpu_mem / 4
    max_chunk_y = max_floats_in_mem / basis.shape[0]
    n_chunks = int(y.shape[0] / max_chunk_y) + 1  
    chunk_y = int(y.shape[0] / n_chunks) + n_chunks

    with torch.no_grad():
        for c in range(n_chunks):
            in_prods = torch.einsum('ik, jk -> ij', y[c * chunk_y : (c + 1) * chunk_y], basis)
            values[c * chunk_y : (c + 1) * chunk_y], indices[c * chunk_y : (c + 1) * chunk_y] = torch.topk(in_prods, non_zeros, dim=1)
            del in_prods

    return indices.to('cpu'), values.to('cpu')

def relative_represent_2(y: torch.Tensor, basis: torch.Tensor, batch_size: int = 100,  k: int = 800, basis_batch_size: int = -1, computing_device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:

    values, indices = torch.zeros((y.shape[0], k)), torch.zeros((y.shape[0], k), dtype=torch.long)

    if basis_batch_size > 0:
        y = y.to("cpu")
        basis = basis.to("cpu")
        for basis_i in range(0, y.size()[0], basis_batch_size):
            basis_batch = basis[basis_i: min(basis_i+basis_batch_size, basis.size()[0])].to(computing_device)
            for i in tqdm(range(0, y.size()[0], batch_size), disable=batch_size==y.size()[0]):
                #in_prods = torch.einsum('ik, jk -> ij', y[c * chunk_y : (c + 1) * chunk_y], basis)
                yy = y[i: min(i+batch_size, y.size()[0])].to(computing_device)
                sim = (1 / (1 + torch.cdist(yy, basis_batch))) 
                top_k = torch.topk(sim, k, dim=1)
                values[i: min(i+batch_size, y.size()[0])], indices[i: min(i+batch_size, y.size()[0])] = top_k[0], top_k[1] + basis_i
                del sim
                del yy
                torch.cuda.empty_cache()
            del basis_batch

    else:
        for i in tqdm(range(0, y.size()[0], batch_size), disable=batch_size==y.size()[0]):
            #in_prods = torch.einsum('ik, jk -> ij', y[c * chunk_y : (c + 1) * chunk_y], basis)
            sim = (1 / (1 + torch.cdist( y[i: min(i+batch_size, y.size()[0])], basis))) 
            values[i: min(i+batch_size, y.size()[0])], indices[i: min(i+batch_size, y.size()[0])] = torch.topk(sim, k, dim=1)
            del sim
            torch.cuda.empty_cache()

    return indices.to('cpu'), values.to('cpu')

def sparsify(i: torch.Tensor, v: torch.Tensor, size: torch.Size) -> torch.sparse.FloatTensor:
    """
    Organize indices and values of n vectors into a single sparse tensor.

    Args:
        i (torch.Tensor): indices of non-zero elements of every vector. Shape: (n_vectors, nonzero elements)
        v (torch.Tensor): values of non-zero elements of every vector. Shape: (n_vectors, nonzero elements)
        size (torch.Size): shape of the output tensor

    Returns:
        torch.sparse.FloatTensor: sparse tensor of shape "size" (n_vectors, zero + nonzero elements)
    """
    flat_dim = len(i.flatten())
    coo_first_row_idxs = torch.div(torch.arange(flat_dim), i.shape[1], rounding_mode='floor')
    stacked_idxs = torch.cat((coo_first_row_idxs.unsqueeze(0), i.flatten().unsqueeze(0)), 0)
    return torch.sparse_coo_tensor(stacked_idxs, v.flatten(), size)


def normalize_sparse(tensor: torch.sparse.FloatTensor, nnz_per_row: int) -> torch.sparse.FloatTensor:
    """
    Normalize a sparse tensor by row.

    Args:
        tensor (torch.sparse.FloatTensor): The sparse tensor to normalize.
        nnz_per_row (int): The number of non-zero elements per row.

    Returns:
        torch.sparse.FloatTensor: The normalized sparse tensor.
    """
    norms = torch.sparse.sum(tensor * tensor, dim=1).to_dense()
    v = tensor._values().clone().detach().reshape(-1, nnz_per_row).t()
    v /= torch.sqrt(norms)
    return torch.sparse_coo_tensor(tensor._indices(), v.t().flatten(), tensor.shape)


def standardize(tensor):
    means = tensor.mean(dim=1, keepdim=True)
    stds = tensor.std(dim=1, keepdim=True)
    return (tensor - means) / stds

def compute_relative_coordinates(embeddings, anchors, k, p=7):

    embeddings = standardize(embeddings)
    anchors = standardize(anchors)
                
    sim = (1 / (1 + torch.cdist(embeddings, anchors)))
    
    #result = torch.zeros(sim.size())
        
    #for i, j in enumerate(torch.argsort(sim, descending=True)[:,:k]):
    #    result[i][j] = p
    indices = [[i, int(j)] for i,j in enumerate(torch.argsort(sim, descending=True)[:,:k])]
    values = [p] * len(indices)
    print(indices)
    print(values)
    print(sim.size())
    
    return torch.sparse_coo_tensor(indices=indices, values=values, size=sim.size())

def compute_self_relative_coordinates(embeddings, anchors, batch_size=1_000, denoise=True, k=800, p=8, device="cpu"):
    
    result = []

    for i in tqdm(range(0, embeddings.size()[0], batch_size), disable=batch_size==embeddings.size()[0]):
        self_relative_coordinates_batch = compute_relative_coordinates(embeddings[i:min(i+batch_size, embeddings.size()[0])], embeddings)
        self_relative_coordinates_batch = self_relative_coordinates_batch.to(device)
        relative_coordinates_vs_anchors = compute_relative_coordinates(self_relative_coordinates_batch, anchors, denoise=denoise, k=k, p=p)
        relative_coordinates_vs_anchors = relative_coordinates_vs_anchors.to("cpu")
        result.append(relative_coordinates_vs_anchors)
    
    return torch.vstack(result)

def evaluate_asif(relative_coordinates_1, relative_coordinates_2, n_of_samples):
    distances_1_to_2 = 1 / (1 + torch.cdist(relative_coordinates_1, relative_coordinates_2))

    # Get the max similarity for each vecotry in relative_coordinates_1
    max_values = distances_1_to_2.max(dim=1, keepdim=True)
    correct = 0

    for sample_index in range(n_of_samples):
        
        # Get indexes of elements with max similarity
        indexes = (distances_1_to_2[sample_index] == max_values.values[sample_index]).nonzero(as_tuple=True)[0]
        

        # Check if the index of the current element is among the elements with the maximum similarity
        if indexes.__contains__(sample_index):
            correct = correct + 1

        #retrieved_elements = itemgetter(*indexes)(elements_2)
        #if elements_2[sample_index] in retrieved_elements:
        #    correct = correct + 1
    
    return correct, int((correct/n_of_samples)*100)


def elbow(X, cluster_sizes, label = "Elbow curve"):
    distorsions = []
    result = {}
    for k in tqdm(cluster_sizes):
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)
        result[k] = kmeans
    fig = plt.figure(figsize=(15, 5))
    plt.plot(cluster_sizes, distorsions)
    plt.xticks(cluster_sizes)
    plt.grid(True)
    plt.title(label)
    return result

data_folder = "data"
computing_device = "cuda:0"
print("loading data")
il = pickle.load(open(f"{data_folder}/lyrics_indexes_similarities.pkl", "rb"))
vl = pickle.load(open(f"{data_folder}/lyrics_values_similarities.pkl", "rb"))
lyrics_embeddings = pickle.load(open(f"{data_folder}/lyrics_embeddings_sbert_roberta.pkl", "rb"))
print("sparsify")
lyrics_sparse = sparsify(il,vl,(lyrics_embeddings.size()[0],lyrics_embeddings.size()[0]))
print("mult")
mult = torch.sparse.mm(lyrics_sparse, lyrics_sparse)
print("sum")
norm = torch.sparse.sum(mult, dim=1)