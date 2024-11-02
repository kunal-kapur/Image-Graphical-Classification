import kornia as K
import torch
from PIL import Image

# Load image
image = Image.open("Animals/dogs/1_0001.jpg").convert("L") 
image = K.image_to_tensor(image, keepdim=True)

# Compute SIFT descriptors
sift = K.feature.SIFTDescriptor(num_ang_bins=8, num_spatial_bins=4)
keypoints, descriptors = sift(image)

print(keypoints)
print(descriptors)