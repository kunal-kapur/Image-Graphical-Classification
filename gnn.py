import torch
from torch.types import Tensor
import torch.nn.functional as F
from typing import Tuple


class GNN(torch.nn.Module):
    def __init__(self, k: int, metric: str):
         """Graph Neural Network

         Args:
            k (int): k-nearest neighbors
            metric (str): metric to compare neighbors
         """
         super(GNN, self).__init__()
         self.k = k

    def nearest_neighbor(self, data: Tensor)->Tensor:
         """Find the nearest neighbor

         Args:
         data (Tensor): (N, 20, 128) tensor of embeddings 

         Returns:
         Tensor, Tensor: (N, )
         """
         N, num_points, D = data.shape  # N: batch size, num_points: 20, D: embedding dim (128)
         
         # Initialize lists to collect edges
         U_list = []
         V_list = []
         
         for i in range(N):
               batch_data = data[i]  # Shape (20, 128), embeddings for one element in the batch
               # Compute pairwise distances within this batch element
               if self.metric == 'euclidean':
                  dist = torch.cdist(batch_data, batch_data, p=2)  # Shape: (20, 20)
               elif self.metric == 'cosine':
                  dist = 1 - F.cosine_similarity(batch_data.unsqueeze(1), batch_data.unsqueeze(0), dim=-1)
               else:
                  raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

               # Get k nearest neighbors (excluding self)
               _, indices = torch.topk(dist, self.k + 1, largest=False)  # takes the closest points
               indices = indices[:, 1:]  # take the given indices

               # Create adjacency list for this batch element
               U = torch.arange(num_points).repeat_interleave(self.k)  # Repeat each point index k times
               V = indices.flatten()  # Flatten the indices tensor to match them

               # Store adjacency list for this batch element
               U_list.append(U)
               V_list.append(V)

         # Concatenate all batch adjacency lists
         U = torch.cat(U_list)
         V = torch.cat(V_list)

         return U, V

    def forward(self, x: Tensor):
       """Forward pass to network

       Args:
           x (Tensor): (N, 20, 128) tensor of embeddings 
       """
       neighbor_tensor = self.nearest_neighbor(x)
       
       

      