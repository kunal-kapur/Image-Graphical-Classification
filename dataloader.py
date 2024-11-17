import os
from torchvision.io import read_image
import numpy as np
# import cv2 as cv
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import kornia as kornia
from torchvision import transforms
from helpers import compute_harris_response, get_harris_points
import pandas as pd
from typing import Union, Tuple
from torch.types import Tensor


class AnimalsDatasetParquet(Dataset):
    """More optimized version of accessing dataset to run inference
    """
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_parquet(path=path)
        self.images = self.df.groupby("path").groups #has coressponding indicies of image
        self.image_keys = list(self.images.keys())
        self.NUM_COLUMNS = len(self.df.columns)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        chosen_indices = self.images[self.image_keys[index]]
        locations = torch.from_numpy(self.df.iloc[chosen_indices, 0:2].values)
        embedding = torch.from_numpy(self.df.iloc[chosen_indices, 2: self.NUM_COLUMNS - 2].values)
        label = ((self.df.iloc[chosen_indices, self.NUM_COLUMNS - 2].unique()))[0]
        path = self.image_keys[index]
        return locations, embedding, label, path



        

class AnimalsDatasetImage(Dataset):
    """Loading dataset by images"""
    def __init__(self, path):
        self.keynet = kornia.feature.KeyNetDetector(pretrained=True, num_features=20)
        self.sift = kornia.feature.SIFTDescriptor(21, 8, 4)
        self.image_paths = []
        self.mapping = {0:"cat", 1: "dog", 2: "snake"}

        for i in os.listdir(os.path.join(path, "cats")):
            self.image_paths.append((os.path.join(path, "cats", i), torch.tensor([1, 0, 0])))

        for i in os.listdir(os.path.join(path, "dogs")):
            self.image_paths.append((os.path.join(path, "dogs", i), torch.tensor([0, 1, 0])))

        for i in os.listdir(os.path.join(path, "snakes")):
            self.image_paths.append((os.path.join(path, "snakes", i), torch.tensor([0, 0, 1])))

    def __len__(self):
        return len(self.image_paths)
    
    def get_label(self, val):
        return self.mapping[torch.argmax(val).item()]
    def _get_keypoints(self, img):
        """
        Extract keypoints from the image.

        Inputs:
            img: h x w tensor.
        Outputs:
            keypoints: N x 2 numpy array.
        """
        keypoints = None
        harrisim = compute_harris_response(img.unsqueeze(dim=0).unsqueeze(dim=0))
        keypoints = (get_harris_points(harrisim, min_distance=15))
        keypoints = torch.tensor(keypoints)
        return keypoints
    
    def _get_descriptors(self, img, keypoints):
        """
        Extract descriptors from the image at the given keypoints.

        Inputs:
            img: h x w tensor.
            keypoints: N x 2 tensor.
        Outputs:
            descriptors: N x D tensor.
        """
        descriptors = None
        patch_size = 32
        half = patch_size // 2

        descriptors = []
        hynet = kornia.feature.HyNet(pretrained=True)
        patches = []
        img = torch.nn.functional.pad(img.unsqueeze(dim=0), pad=(half, half, half, half), mode='reflect').squeeze(dim=0)
        
        for y, x in keypoints:
            y, x = y + half, x + half # account for the padding
            y, x = int(y.item()), int(x.item())
            patch = img[y-half:y+half, x-half:x+half]
            patches.append(patch)
        descriptors = hynet(torch.stack(tuple(patches), dim=0).unsqueeze(dim=1))
        return descriptors

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, str]:
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path)
        # Define the transformation
        grayscale_transform = transforms.Grayscale()
        grayscale_image = grayscale_transform(image)
        tensor_image = transforms.ToTensor()(grayscale_image).unsqueeze(dim=0)
        keypoints = self.keynet.forward(tensor_image)[0]
        pixel_locations = keypoints[:, :, :2, 2].squeeze(dim=0) # take x, y coordinates
        desc = self._get_descriptors(img=tensor_image.squeeze(dim=0).squeeze(dim=0), 
                                     keypoints=pixel_locations)
        return  pixel_locations, desc, label, img_path