import torch
from torch.types import Tensor
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv  # Import a GNN layer
from torch.nn import Linear
import matplotlib.pyplot as plt
import numpy as np


class GNN(torch.nn.Module):
   def __init__(self, k: int, metric: str='euclidean'):
      """Graph Neural Network

      Args:
         k (int): k-nearest neighbors
         metric (str): metric to compare neighbors
      """
      super(GNN, self).__init__()
      EMBEDDING_SIZE = 128
      self.k = k
      self.metric=metric
      self.conv1 = GCNConv(EMBEDDING_SIZE, 256, node_dim=1)
      self.conv2 = GCNConv(256, 512, node_dim=1)
      self.conv3 = GCNConv(512, 128, node_dim=1)
      self.conv_layers = [self.conv1, self.conv2, self.conv3]

      self.linear4 = Linear(128, 64)
      self.linear5 = Linear(64, 3)
         

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
      U = torch.stack(U_list, dim=0)
      V = torch.stack(V_list, dim=0)
      val = torch.stack([U, V], dim=1)

      return val
   

   def plot_image(self, image_path, edges, coordinates):
      image = plt.imread(image_path)
      y, x = coordinates[:, 0], coordinates[:, 1]
      fig, ax = plt.subplots()
      ax.imshow(image)
      ax.scatter(x.numpy(), y.numpy(), edgecolors='blue',  # Box color
      facecolors='none',  # Empty box
      marker='s', s=30)
      for edge in edges.T:
         origin, dest = coordinates[edge[0]], coordinates[edge[1]]
         ax.plot([origin[1], dest[1]], [origin[0], dest[0]], color='red', linewidth=0.1, label='Line 1')
         ax.annotate(
        '',  # No text
        xy=(dest[1], dest[0]),  # Destination point
        xytext=(origin[1], origin[0]),  # Origin point
        arrowprops=dict(arrowstyle='->', color='red', lw=1)
    )
      # ax.plot(x_starts.numpy(), y_starts.numpy(), color='red', linewidth=2)
      plt.show()



   def forward(self, x: Tensor, intermediate_graphs:bool=False, coordinates: Tensor=None, image_path: str=""):
      """Forward pass to network

      Args:
         x (Tensor): (N, 20, 128) tensor of embeddings 
      """
      
      BATCH, _, _ = x.shape


      if intermediate_graphs is True and coordinates is None:
         raise ValueError("Please add coordinates")
      
      if intermediate_graphs is True and BATCH > 1:
         raise ValueError("Batch processing of over 1 not supported for graphing")


      for i in range(len(self.conv_layers)):
         N, num_points, D = x.shape
         edge_index = (self.nearest_neighbor(x))
         if intermediate_graphs is True:
            self.plot_image(image_path=image_path, edges=edge_index.squeeze(dim=0), coordinates=coordinates.squeeze(dim=0))
         data = Data(x=x, edge_index=edge_index)
         x = self.conv_layers[i](data.x, data.edge_index.squeeze(dim=0))
         x = F.relu(x)
      x = x.mean(dim=1)
      x = self.linear4(x)
      x = F.relu(x)
      x = self.linear5(x)
      return F.softmax(x, dim=1)
