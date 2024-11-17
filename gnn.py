import torch
from torch.types import Tensor
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv  # Import a GNN layer
from torch.nn import Linear


class GNN(torch.nn.Module):
   def __init__(self, k: int, num_nodes:int, metric: str='euclidean'):
      """Graph Neural Network

      Args:
         k (int): k-nearest neighbors
         metric (str): metric to compare neighbors
      """
      super(GNN, self).__init__()
      EMBEDDING_SIZE = 128
      self.k = k
      self.metric=metric
      self.num_nodes = num_nodes
      self.conv1 = GCNConv(EMBEDDING_SIZE, 300, node_dim=1)
      self.conv2 = GCNConv(300, 100, node_dim=1)
      self.conv3 = GCNConv(100, 50, node_dim=1)
      self.conv_layers = [self.conv1, self.conv2, self.conv3]
      self.linear4 = Linear(num_nodes * 50, 100)
      self.linear5 = Linear(100, 3)
         

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
      val = torch.stack([U, V], dim=0)
      if len(val.shape) == 2:
         val = val.unsqueeze(dim=0)
      return val



   def forward(self, x: Tensor):
      """Forward pass to network

      Args:
         x (Tensor): (N, 20, 128) tensor of embeddings 
      """
      BATCH, _, _ = x.shape
      for i in range(len(self.conv_layers)):
         N, num_points, D = x.shape
         edge_index = (self.nearest_neighbor(x))
         data = Data(x=x, edge_index=edge_index)
         x = self.conv_layers[i](data.x, data.edge_index.squeeze(dim=0))
         x = F.relu(x)
      x = x.view(BATCH, -1)
      x = self.linear4(x)
      x = F.relu(x)
      x = self.linear5(x)
      return F.softmax(x, dim=1)
