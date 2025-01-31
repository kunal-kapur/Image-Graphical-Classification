from dataloader import AnimalsDatasetParquet, AnimalsDatasetImage
from gnn import GNN
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from image_to_parquet import animals_parquet
from torch.optim.lr_scheduler import StepLR


import argparse



torch.manual_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("--distance", default=15)   
parser.add_argument('-e', "--epochs", default=10)    
parser.add_argument('-k', "--neighbors", default=3)    
parser.add_argument('-l', "--lr", default=0.001)
parser.add_argument('-s', "--schedule", default=6)      
parser.add_argument('-c', "--classified", default=False)    
parser.add_argument('-n', "--nodes", default=20)
parser.add_argument('--data', default="Animals")    

args = parser.parse_args()
K = int(args.neighbors)
model = GNN(k=K)

print(args)

LR = float(args.lr)
BATCH_SIZE = 1
EPOCHS = int(args.epochs)
DIST = int(args.distance)
NODES = int(args.nodes)
DATA_PATH = str(args.data)

torch.manual_seed(0)


CLASSIFIED = bool(args.classified)
parquet_path = f"animals_d{DIST}_nodes{NODES}_classified{CLASSIFIED}.parquet"

if not os.path.exists(parquet_path):
    print(f"Creating {parquet_path}")
    animals_parquet(DIST, nodes=NODES, classified=CLASSIFIED, data_path=DATA_PATH)

data = AnimalsDatasetParquet(parquet_path)


train, validation, test = random_split(data, lengths=[0.7, 0.15, 0.15])

dataloader = DataLoader(train, batch_size=BATCH_SIZE)

dataloader_validation = DataLoader(validation, batch_size=BATCH_SIZE)


loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# use a scheduler to get stable training since batch size is 1
SCHEDULE = int(args.schedule)
scheduler = StepLR(optimizer, step_size=SCHEDULE)

model.train()

train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []
for epoch in tqdm(range(EPOCHS)):
    tot_correct_train = 0
    tot_correct_val = 0
    model.train()
    tot_c1, tot_c2, tot_c3 = 0, 0, 0
    correct_c1, correct_c2, correct_c3 = 0, 0, 0
    train_loss, val_loss = 0, 0
    for i, val in tqdm(enumerate(dataloader)):
        coords, inputs, targets, paths = val
        targets = data.map_label(targets) # make labels into integers

        tot_c1, tot_c2, tot_c3 = torch.sum(targets == 0), torch.sum(targets == 1), torch.sum(targets == 2)
        out = model.forward(inputs)
        loss = loss_func(out, targets)
        train_loss += loss
        tot_correct_train += torch.sum(out.argmax(dim=1) == targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for i, val in tqdm(enumerate(dataloader_validation)):
            coords, inputs, targets, paths = val
            targets = data.map_label(targets) # make labels into integers
            out = model.forward(inputs)
            loss = loss_func(out, targets)
            val_loss += loss
            tot_correct_val += torch.sum(out.argmax(dim=1) == targets)
    print(tot_correct_val, len(validation))
    print("Train accuracy: ", (tot_correct_train / len(train)).item())
    print("Validation accuracy", (tot_correct_val / len(validation)).item())
    train_acc_list.append((tot_correct_train / len(train)).item())
    val_acc_list.append((tot_correct_val / len(validation)).item())

    train_loss_list.append((train_loss / len(train)).item())
    val_loss_list.append((val_loss / len(validation)).item())

    scheduler.step()


tot_correct_test = 0

class_totals = {}
class_correct = {}
num_classes = 3
confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
with torch.no_grad():
    for i, val in tqdm(enumerate(test)):
        coords, inputs, targets, paths = val
        targets = data.map_label((targets,))  # Make labels into integers
        unique_classes = torch.unique(targets).tolist()

        # Ensure all classes are in the dictionaries
        for class_name in unique_classes:
            if class_name not in class_totals:
                class_totals[class_name] = 0
                class_correct[class_name] = 0

        for class_name in unique_classes:
            class_totals[class_name] += torch.sum(targets == class_name).item()

        out = model.forward(torch.unsqueeze(inputs, dim=0))
        preds = out.argmax(dim=1)

        # Update confusion matrix
        for t, p in zip(targets, preds):
            confusion_matrix[t.item(), p.item()] += 1

        # Update class-wise correct counts
        for cls in unique_classes:
            class_correct[cls] += torch.sum((preds == cls) & (targets == cls)).item()

        # Update total correct predictions
        tot_correct_test += torch.sum(preds == targets).item()

# Convert confusion matrix to numpy for easier visualization (optional)
confusion_matrix_np = confusion_matrix.numpy()


test_acc = (tot_correct_test / len(test))

# total test accuracy 
print("Test accuracy ", test_acc)

# Per-class accuracy
print("Per-Class Accuracy:")
for class_name in sorted(class_totals.keys()):
    total = class_totals[class_name]
    correct = class_correct[class_name]
    acc = correct / total if total > 0 else 0
    print(correct, total)
    print(f"Class {class_name} Accuracy: {acc * 100:.2f}%")


params = f"classified{CLASSIFIED}_nodes{NODES}_dist{DIST}_k{K}_epochs{EPOCHS}_schedule{SCHEDULE}"
PATH = f"results/{params}"
if not os.path.exists(PATH):
    os.makedirs(PATH)

x = np.arange(EPOCHS)
plt.figure()
plt.plot(x, train_loss_list, label="training")
plt.plot(x, val_loss_list, label="validation")
plt.title(f'Training Loss {params}')
plt.legend()
plt.savefig(os.path.join(PATH, 'training_curve_loss.png'))  # Save the plot

plt.figure()
plt.plot(x, train_acc_list, label="training")
plt.plot(x, val_acc_list, label="validation")
plt.title(f'Training Accuracy {params}')
plt.legend()
plt.savefig(os.path.join(PATH, 'training_curve_accuracy.png'))  # Save the plot


plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix_np, interpolation='nearest', cmap='Blues')
plt.colorbar()

# Add labels, title, and axis ticks
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Add grid lines for better readability
plt.grid(False)

# Annotate the confusion matrix cells
num_classes = confusion_matrix_np.shape[0]
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{confusion_matrix_np[i, j]}',
                 ha='center', va='center',
                 color='white' if confusion_matrix_np[i, j] > confusion_matrix_np.max() / 2 else 'black')

# Set tick marks to match classes
plt.xticks(range(num_classes), [f'Class {i}' for i in range(num_classes)], rotation=45)
plt.yticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])

plt.tight_layout()
plt.savefig(os.path.join(PATH, 'confusion matrix'))




with open(os.path.join(PATH, "test_acc.txt"), 'w') as f:
    f.write(f"Test Accuracy: {test_acc}\n")
    for class_name in sorted(class_totals.keys()):
        total = class_totals[class_name]
        correct = class_correct[class_name]
        acc = correct / total if total > 0 else 0
        f.write(f"Class {class_name} Accuracy: {acc * 100:.2f}%\n")

torch.save(model.state_dict(), os.path.join(PATH, 'model_weights.pth'))

with open(os.path.join(PATH, "training.txt"), 'w') as f:
    f.write(f"train loss: {train_loss_list}\n")
    f.write(f"validation loss: {val_loss_list}\n")
    f.write(f"train acc: {train_acc_list}\n")
    f.write(f"validation acc acc: {val_acc_list}")




