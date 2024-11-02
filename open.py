import kornia as kornia
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# Load image
# image = Image.open("Animals/dogs/1_0001.jpg").convert("L") 
image = Image.open("chess.jpg")
# Define the transformation
# grayscale_transform = transforms.Grayscale()
# grayscale_image = grayscale_transform(image)

tensor_image = (transforms.functional.pil_to_tensor(image).float() / 255).unsqueeze(dim=0)
# # tensor_image = transforms.ToTensor()(grayscale_image).unsqueeze(dim=0)
# tensor_image = (kornia.utils.image_to_tensor(np.array(image), False).float() / 255.)
tensor_image = kornia.color.rgb_to_grayscale(tensor_image)
print("size", tensor_image.shape)


keynet = kornia.feature.KeyNetDetector(pretrained=True, num_features=20)
# Detect keypoints
keypoints = keynet.forward(tensor_image)[0]  # Shape: (1, N, 2, 3)
print("Got keypoints")

# Extract (x, y) coordinates from the affine transformation
pixel_locations = keypoints[:, :, :2, 2]  # (1, N, 2), with (x, y) in the last dimension
print(pixel_locations.shape)
x_coords = pixel_locations[0, :, 0].cpu().numpy()
y_coords = pixel_locations[0, :, 1].cpu().numpy()

print("Plotting")
# Plot the image
plt.imshow(tensor_image[0, 0].cpu(), cmap='gray')
plt.scatter(x_coords, y_coords, c='red', s=15, marker='x')  # Plot keypoints

plt.title("Keypoints detected by KeyNetDetector")
plt.axis("off")
plt.show()