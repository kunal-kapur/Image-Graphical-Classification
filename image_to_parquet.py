from dataloader import AnimalsDatasetImage
import pyarrow as pa
import pyarrow.parquet as pq
from torch.types import Tensor
import torch
import pandas as pd

data = AnimalsDatasetImage("Animals")

print(len(data))
names=['y', 'x']
for i in range(128):
    names.append(f"feature_{i}")

label_list = []
path_list = []
dimension_arr = torch.tensor([])
for i, sample in enumerate(data):
    loc: Tensor
    description: Tensor
    label: Tensor
    path: str
    loc, description, label, path = sample
    combined = torch.concat((loc, description), dim=1)
    dimension_arr = torch.concat((dimension_arr, combined))
    path_list += [path] * combined.shape[0]
    label_list += [data.get_label(label)] * combined.shape[0]

df = pd.DataFrame(data=dimension_arr.detach().numpy(),  columns=names)
df['label'] = label_list
df['path'] = path_list

df.to_parquet('animals.parquet')

