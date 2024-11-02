import os
from torchvision.io import read_image
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
import os
import torch


class CustomImageDataset(Dataset):
    def __init__(self, path):
        CWD = os.getcwd()
        self.sift = cv.SIFT_create()
        self.image_paths = []
        self.mapping = {0:"cat", 1: "dog", 2: "snake"}

        for i in os.listdir(os.path.join(path, "Animals", "cats")):
            self.image_paths.append((os.path.join(CWD, "cats", i), torch.tensor([1, 0, 0])))

        for i in os.listdir(os.path.join(path, "Animals", "dogs")):
            self.image_paths.append((os.path.join(CWD, "dogs", i), torch.tensor([0, 1, 0])))

        for i in os.listdir(os.path.join(path, "Animals", "snakes")):
            self.image_paths.append(os.path.join(CWD, "snakes", torch.tensor([0, 0, 1])))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths
        img = cv.imread('/Users/kunalkapur/Workspace/cs593-proj/Animals/dogs/1_0001.jpg')
        return img, label