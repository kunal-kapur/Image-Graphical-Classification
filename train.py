from dataloader import AnimalsDatasetParquet
from gnn import GNN
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm



torch.manual_seed(10)

# num nodes is hard coded to be 20, can't change without change in parquet
model = GNN(k=5, num_nodes=20)

LR = 0.001
BATCH_SIZE = 1
EPOCHS = 10

torch.manual_seed(0)
data = AnimalsDatasetParquet("animals.parquet")


print(len(data))
train, val, _ = random_split(data, lengths=[1000, 500, len(data) - 1500])

print(len(data))

dataloader = DataLoader(train, batch_size=BATCH_SIZE)

dataloader_val = DataLoader(val, batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


model.train()
for epoch in tqdm(range(EPOCHS)):
    tot_correct_train = 0
    tot_correct_val = 0
    model.train()
    for i, val in tqdm(enumerate(dataloader)):
        if i < 110:
            continue
        coords, inputs, targets, paths = val

        targets = data.map_label(targets) # make labels into integers


        out = model.forward(inputs, intermediate_graphs=True, coordinates=coords, image_path=paths[0])

        loss = loss_func(out, targets)
        tot_correct_train += torch.sum(out.argmax(dim=1) == targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for i, val in tqdm(enumerate(dataloader)):
            coords, inputs, targets, paths = val

            targets = data.map_label(targets) # make labels into integers
            out = model.forward(inputs)
            loss = loss_func(out, targets)
            tot_correct_val += torch.sum(out.argmax(dim=1) == targets)

    
    print("train", tot_correct_train / 1000)
    print("val", tot_correct_val / 1000)