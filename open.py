from gnn import GNN
import torch
import os
from dataloader import AnimalsDatasetParquet


# Create an instance of your model
model = GNN(k=3) 

# Load the state dictionary from the saved file
model.load_state_dict(torch.load(os.path.join("results", "animals_dist15_k0_epochs18_schedule6",
                                             "model_weights.pth"))) 

dist = 15
data = AnimalsDatasetParquet(f"animals_{dist}.parquet")

# coords, inputs, targets, paths = data[0]

for i in range(0, len(data) - 1000, 50):
    coords, inputs, targets, paths = data[i]
    model.forward(torch.unsqueeze(input=inputs, dim=0), intermediate_graphs=True, coordinates=coords, image_path=paths)


# 10742, 10794, 0053