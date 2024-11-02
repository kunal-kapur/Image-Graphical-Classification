from dataloader import AnimalsDataset

data = AnimalsDataset("Animals")

print(len(data))

for i, sample in enumerate(data):
    print(i, sample)
    break
