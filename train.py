from dataloader import AnimalsDatasetParquet
from gnn import GNN
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm



# num nodes is hard coded to be 20, can't change without change in parquet
model = GNN(k=5, num_nodes=20)

LR = 0.001
BATCH_SIZE = 16
EPOCHS = 10

torch.manual_seed(0)
data = AnimalsDatasetParquet("animals.parquet")


print(len(data))
train, _ = random_split(data, lengths=[500, len(data) - 500])

print(len(data))

dataloader = DataLoader(train, batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



model.train()
for epoch in range(EPOCHS):
    tot_correct = 0
    for i, val in tqdm(enumerate(dataloader)):
        coords, inputs, targets, paths = val

        targets = data.map_label(targets) # make labels into integers

        out = model.forward(inputs)

        loss = loss_func(out, targets)
        tot_correct += torch.sum(out.argmax(dim=1) == targets)

        with torch.no_grad():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(tot_correct / 500)