import kornia as kornia
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# Load image
# image = Image.open("Animals/dogs/1_0001.jpg").convert("L") 
image = Image.open("Animals/dogs/1_0002.jpg")
# Define the transformation
grayscale_transform = transforms.Grayscale()
grayscale_image = grayscale_transform(image)

tensor_image = transforms.ToTensor()(grayscale_image).unsqueeze(dim=0)
print("size", tensor_image.shape)
# image = K.image_to_tensor(image, keepdim=True)

# Compute SIFT descriptors
# sift = kornia.feature.DenseSIFTDescriptor()
# print(tensor_image.shape)
# val = sift(tensor_image.unsqueeze(dim=0))
# print(val.shape)
# # print(keypoints)
# # print(descriptors)
# print(kornia.feature.harris_response(input, 0.04))
# input = torch.tensor([[[
#    [0., 0., 0., 0., 0., 0., 0.],
#    [0., 1., 1., 1., 1., 1., 0.],
#    [0., 1., 1., 1., 1., 1., 0.],
#    [0., 1., 1., 1., 1., 1., 0.],
#    [0., 1., 1., 1., 1., 1., 0.],
#    [0., 1., 1., 1., 1., 1., 0.],
#    [0., 0., 0., 0., 0., 0., 0.],
# ]]])  # 1x1x7x7
# # compute the response map
# print(tensor_image.shape)
# sift.

detector = kornia.feature.CornerHarris(k=0.4)
response = detector.forward(input=tensor_image).squeeze(dim=0).squeeze(dim=0)
H, W = response.shape
indices = torch.argsort(response.flatten())
best = torch.argsort(indices)[0:20]
vals = torch.unravel_index(best, (H, W))
matches = torch.stack(tensors=vals, dim=0)
print(matches.shape)

# Simulate the matches variable (replace this with your actual matches)
# matches should be a tensor or array with shape (2, 50)
# Assume matches[0, :] are x-coordinates and matches[1, :] are y-coordinates
x_coords = matches[0, :].cpu().numpy()  # Convert to numpy if in tensor format
y_coords = matches[1, :].cpu().numpy()

# Plot the image
plt.imshow(tensor_image[0, 0].cpu(), cmap='gray')
plt.scatter(y_coords, x_coords, c='red', s=15, marker='x')  # Plot keypoints

plt.title("Keypoints")
plt.axis("off")
plt.show()

# keynet = kornia.feature.KeyNetDetector(pretrained=True, num_features=40)
# # Detect keypoints
# keypoints = keynet.forward(tensor_image)[0]  # Shape: (1, N, 2, 3)

# # Extract (x, y) coordinates from the affine transformation
# pixel_locations = keypoints[:, :, :2, 2]  # (1, N, 2), with (x, y) in the last dimension
# print(pixel_locations.shape)
# x_coords = pixel_locations[0, :, 0].cpu().numpy()
# y_coords = pixel_locations[0, :, 1].cpu().numpy()

# # Plot the image
# plt.imshow(tensor_image[0, 0].cpu(), cmap='gray')
# plt.scatter(x_coords, y_coords, c='red', s=15, marker='x')  # Plot keypoints

# plt.title("Keypoints detected by KeyNetDetector")
# plt.axis("off")
# plt.show()