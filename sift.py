import numpy as np
import cv2 as cv

# Load the image
img = cv.imread('/Users/kunalkapur/Workspace/cs593-proj/Animals/dogs/1_0001.jpg')

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create a SIFT detector object
sift = cv.SIFT_create()

# Detect keypoints in the grayscale image
kp = sift.detect(gray, None)
# kp, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
img_with_keypoints = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save the image with keypoints
cv.imwrite('sift_keypoints.jpg', img_with_keypoints)

# Extract keypoint locations and sizes
keypoint_locations = np.array([kp[i].pt for i in range(len(kp))])  # Extract (x, y) coordinates
keypoint_sizes = np.array([kp[i].size for i in range(len(kp))])  # Sizes of keypoints

# Print keypoint locations and sizes
# print(descriptors.shape)
print(keypoint_locations.shape)
# print("Keypoint Locations (x, y):")
# print(keypoint_locations)
# print("Keypoint Sizes (diameter):")
# print(keypoint_sizes)
