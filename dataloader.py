import os
from torchvision.io import read_image
import numpy as np
# import cv2 as cv
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import kornia as kornia
from kornia.feature.scale_space_detector import get_default_detector_config
from torchvision import transforms
from helpers import compute_harris_response, get_harris_points
import pandas as pd
from typing import Union, Tuple
from torch.types import Tensor
from object_detector import MLP


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
    
    def map_label(self, input):
        map = {"dog": 0, "cat": 1, "snake": 2}
        num_classes = len(map)

        if isinstance(input, str):
            index = map.get(input, None)
            if index is None:
                raise ValueError(f"Invalid input: {input} is not a recognized label.")
            one_hot = torch.zeros(num_classes)
            one_hot[index] = 1
            return one_hot

        elif isinstance(input, tuple):
            new_vals = []
            for i in input:
                new_vals.append(map[i])
            return torch.tensor(new_vals)
    
        else:
            raise TypeError("Input must be a string or a 1-dimensional tensor.")

            
    
    def __getitem__(self, index):
        chosen_indices = self.images[self.image_keys[index]]
        locations = torch.from_numpy(self.df.iloc[chosen_indices, 0:2].values)
        embedding = torch.from_numpy(self.df.iloc[chosen_indices, 2: self.NUM_COLUMNS - 2].values)
        label = ((self.df.iloc[chosen_indices, self.NUM_COLUMNS - 2].unique()))[0]
        path = self.image_keys[index]
        
        return locations, embedding, label, path

class AnimalsDatasetImage(Dataset):
    """Loading dataset by images"""
    def __init__(self, path, distance=10, classify=False):
        self.distance = distance
        self.image_paths = []
        self.mapping = {0:"cat", 1: "dog", 2: "snake"}

        for i in os.listdir(os.path.join(path, "cats")):
            self.image_paths.append((os.path.join(path, "cats", i), 0))

        for i in os.listdir(os.path.join(path, "dogs")):
            self.image_paths.append((os.path.join(path, "dogs", i), 1))

        for i in os.listdir(os.path.join(path, "snakes")):
            self.image_paths.append((os.path.join(path, "snakes", i), 2))

        self.model = None
        if classify is True:
            self.model = MLP()
            self.model.load_state_dict(torch.load(os.path.join("binary_results",
                                                "model_weights.pth"))) 

    def __len__(self):
        return len(self.image_paths)
    
    def get_label(self, val):
        return self.mapping[torch.argmax(val).item()]
    
    def map_label(self, input):
        map = {"dog": 0, "cat": 1, "snake": 2}
        num_classes = len(map)

        if isinstance(input, str):
            index = map.get(input, None)
            if index is None:
                raise ValueError(f"Invalid input: {input} is not a recognized label.")
            one_hot = torch.zeros(num_classes)
            one_hot[index] = 1
            return one_hot

        elif isinstance(input, tuple):
            new_vals = []
            for i in input:
                new_vals.append(map[i])
            return torch.tensor(new_vals)
    
        else:
            raise TypeError("Input must be a string or a 1-dimensional tensor.")


    
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
        tensor_image = transforms.ToTensor()(grayscale_image).squeeze(dim=0)
        harrisim = compute_harris_response(tensor_image.squeeze(dim=0).squeeze(dim=0))
        keypoints = (get_harris_points(harrisim, min_distance=self.distance))
        if len(keypoints) <= 5:
            return None, None, None, None
        keypoints = torch.tensor(keypoints)
        pixel_locations = keypoints # take x, y coordinates
        desc = self._get_descriptors(img=tensor_image.squeeze(dim=0).squeeze(dim=0), 
                                     keypoints=pixel_locations)

        # Apply the mask to filter pixel_locations, desc, and img_path
        if self.model is not None:
            # Get classification results
            classifications = self.model(desc)  # Assuming it returns an n-sized tensor of 0s and 1s
            # Mask for classifications where the result is 1
            mask = classifications == 1  # Create a boolean mask
            pixel_locations = pixel_locations[mask]
            desc = desc[mask]
            img_path = [img_path[i] for i in range(len(img_path)) if mask[i]]
            label = [label[i] for i in range(len(img_path)) if mask[i]]
        return  pixel_locations, desc, label, img_path
    


class BinaryDatasetParquet(Dataset):
    """More optimized version of accessing dataset to run inference
    """
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_parquet(path=path)
    
    def __len__(self):
        return len(self.df)
            
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.df.iloc[index, 0:128].values)
        y = self.df.loc[index, 'label']
        x.requires_grad_()
        return x, y

class BinaryDataImage(Dataset):
    """Loading dataset by images"""
    def __init__(self, path, distance=10):
        
        self.distance = distance
        self.image_paths = []

        for i in os.listdir(os.path.join(path, "cats")):
            self.image_paths.append((os.path.join(path, "cats", i), 1))

        for i in os.listdir(os.path.join(path, "dogs")):
            self.image_paths.append((os.path.join(path, "dogs", i), 1))

        for i in os.listdir(os.path.join(path, "snakes")):
            self.image_paths.append((os.path.join(path, "snakes", i), 1))

        for i in os.listdir(os.path.join("images")):
            self.image_paths.append((os.path.join("images", i), 0))


    def __len__(self):
        return len(self.image_paths)
    
    def get_label(self, val):
        return self.mapping[torch.argmax(val).item()]
     
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
        tensor_image = transforms.ToTensor()(grayscale_image).squeeze(dim=0)
        harrisim = compute_harris_response(tensor_image.squeeze(dim=0).squeeze(dim=0))
        keypoints = (get_harris_points(harrisim, min_distance=self.distance))
        if len(keypoints) < 5:
            return None, None
        keypoints = torch.tensor(keypoints)
        pixel_locations = keypoints # take x, y coordinates
        desc = self._get_descriptors(img=tensor_image.squeeze(dim=0).squeeze(dim=0), 
                                     keypoints=pixel_locations)
        return  desc, label
    



    


