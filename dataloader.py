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


class AnimalsDataset(Dataset):
    def __init__(self, path):
        self.sift = kornia.feature.SIFTDescriptor(21, 8, 4)
        self.image_paths = []
        self.mapping = {0:"cat", 1: "dog", 2: "snake"}

        for i in os.listdir(os.path.join(path, "cats")):
            self.image_paths.append((os.path.join(path, "cats", i), torch.tensor([1, 0, 0])))

        for i in os.listdir(os.path.join(path, "dogs")):
            self.image_paths.append((os.path.join(path, "dogs", i), torch.tensor([0, 1, 0])))

        for i in os.listdir(os.path.join(path, "snakes")):
            self.image_paths.append((os.path.join(path, "snakes"), torch.tensor([0, 0, 1])))

    def __len__(self):
        return len(self.image_paths)
    
    def _get_keypoints(self, img):
        """
        Extract keypoints from the image.

        Inputs:
            img: h x w tensor.
        Outputs:
            keypoints: N x 2 numpy array.
        """
        keypoints = None
        print(img.shape)
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
        img = torch.nn.functional.pad(img, (half, half, half, half), mode='constant', value=0)
        descriptors = []
        hynet = kornia.feature.HyNet(pretrained=True)
        patches = []
        for y, x in keypoints:
            y, x = y + half, x + half
            patch = img[y-half:y+half, x-half:x+half]
            patches.append(patch)
        descriptors = hynet(torch.stack(tuple(patches), dim=0).unsqueeze(dim=1))
        return descriptors

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path)
        # Define the transformation
        grayscale_transform = transforms.Grayscale()
        grayscale_image = grayscale_transform(image)
        tensor_image = transforms.ToTensor()(grayscale_image).unsqueeze(dim=0)
        keypoints = self._get_keypoints(tensor_image)
        desc = self._get_descriptors(img=tensor_image, keypoints=keypoints)
        return  desc, label