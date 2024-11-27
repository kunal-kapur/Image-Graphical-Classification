import torch 

from dataloader import BinaryDataImage, BinaryDatasetParquet
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from object_detector import MLP
import numpy as np
from matplotlib import pyplot as plt

torch.manual_seed(0)

# Initialize PyArrow table writer
parquet_file = f'objects.parquet'

if not os.path.exists(parquet_file):
    data = BinaryDataImage(path="Animals", distance=25)
    names = [f"feature_{i}" for i in range(128)]
    writer = None
    for x, y in tqdm(data):
        if x is None or y is None:
            continue
        # Convert DataFrame to PyArrow table and write in batches
        if y == 1:
            x = x[0:5]
        else:
            x = x[0:10]
        # Create DataFrame for the current batch
        df = pd.DataFrame(data=x.detach().numpy(), columns=names)
        labels = [y] * x.shape[0]
        df['label'] = labels
        # Convert DataFrame to PyArrow table and write in batches
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()

    print(f"Data written to {parquet_file} in chunks.")

data = BinaryDatasetParquet("objects.parquet")

model = MLP()

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 15

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train, validation, test = random_split(data, lengths=[0.7, 0.15, 0.15])

dataloader_train = DataLoader(train, batch_size=BATCH_SIZE)
dataloader_val = DataLoader(validation, batch_size=BATCH_SIZE)

train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []
for i in range(EPOCHS):
    tot_correct_train = 0
    tot_correct_val = 0
    model.train()
    train_loss, val_loss = 0, 0
    for i, val in tqdm(enumerate(dataloader_train)):
        x, y = val
        out = model.forward(x.float())
        loss = loss_func(out, y)
        train_loss += loss
        tot_correct_train += torch.sum(out.argmax(dim=1) == y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for i, val in tqdm(enumerate(dataloader_val)):
            x, y = val
            out = model.forward(x.float())
            loss = loss_func(out, y)
            val_loss += loss
            tot_correct_val += torch.sum(out.argmax(dim=1) == y)
    print("Train accuracy: ", (tot_correct_train / len(train)).item())
    print("Validation accuracy", (tot_correct_val / len(validation)).item())
    train_acc_list.append((tot_correct_train / len(train)).item())
    val_acc_list.append((tot_correct_val / len(validation)).item())

    train_loss_list.append((train_loss / len(train)).item())
    val_loss_list.append((val_loss / len(validation)).item())


tot_correct_test = 0

with torch.no_grad():
    for i, val in tqdm(enumerate(test)):
        x, y = val
        out = model.forward(torch.unsqueeze(x.float(), dim=0))
        preds = out.argmax(dim=1)
        # Update total correct predictions
        tot_correct_test += torch.sum(preds == y).item()

test_acc = (tot_correct_test / len(test))

# total test accuracy 
print("Test accuracy ", test_acc)


PATH = "binary_results"
os.makedirs(PATH, exist_ok=True)
x = np.arange(EPOCHS)
plt.figure()
plt.plot(x, train_loss_list, label="training")
plt.plot(x, val_loss_list, label="validation")
plt.title(f'Training Loss')
plt.legend()
plt.savefig(os.path.join(PATH, 'training_curve_loss.png'))  # Save the plot

plt.figure()
plt.plot(x, train_acc_list, label="training")
plt.plot(x, val_acc_list, label="validation")
plt.title(f'Training Accuracy')
plt.legend()
plt.savefig(os.path.join(PATH, 'training_curve_accuracy.png'))  # Save the plot

torch.save(model.state_dict(), os.path.join(PATH, 'model_weights.pth'))


