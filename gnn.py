import torch
from torch.types import Tensor
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # Import a GNN layer


class GNN(torch.nn.Module):
   def __init__(self, k: int, metric: str='euclidean'):
      """Graph Neural Network

      Args:
         k (int): k-nearest neighbors
         metric (str): metric to compare neighbors
      """
      super(GNN, self).__init__()
      self.k = k
      self.metric=metric
      self.conv1 = GCNConv(128, 300)
      self.conv2 = GCNConv(128, 300)
      self.conv3 = GCNConv(300, 500)
      self.conv4 = GCNConv(500, 100)
         

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
            indices = indices[:, 1:]

            # repeat each element k times to create edge
            U = torch.arange(num_points).repeat_interleave(self.k)
            V = indices.flatten()

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
      N, num_points, D = x.shape
      U, V = self.nearest_neighbor(x)
      print(U)
      print(V)

      # Reshape x to (N * num_points, D) for graph layers
      x = x.view(N * num_points, D)
       
       
       

      