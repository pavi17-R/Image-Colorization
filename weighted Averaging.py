import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

# Load the colorized images from both models (Autoencoder and U-Net)
output_autoencoder = cv2.imread(r"C:\Users\pavit\OneDrive\Pictures\pot1.png")  # Autoencoder colorized image
output_unet = cv2.imread(r"C:\Users\pavit\OneDrive\Pictures\pot2.png")  # U-Net colorized image
ground_truth = cv2.imread(r"C:\Users\pavit\OneDrive\Pictures\Screenshots\Screenshot 2024-12-11 101024.png")  # Ground truth image

# Check if all images were loaded correctly
if output_autoencoder is None or output_unet is None or ground_truth is None:
    print("Error loading one of the images. Please check the file paths.")
    exit()

# Ensure both colorized images and ground truth have the same dimensions (resize if necessary)
if output_autoencoder.shape != output_unet.shape or output_autoencoder.shape != ground_truth.shape:
    output_unet = cv2.resize(output_unet, (output_autoencoder.shape[1], output_autoencoder.shape[0]))
    ground_truth = cv2.resize(ground_truth, (output_autoencoder.shape[1], output_autoencoder.shape[0]))

# Define the weights for each model's output
weight_autoencoder = 0.6
weight_unet = 0.4

# Perform weighted averaging (pixel-wise) to combine the colorized images
output_combined = weight_autoencoder * output_autoencoder + weight_unet * output_unet

# Ensure the values are within the valid image range [0, 255] for uint8 images
output_combined = np.clip(output_combined, 0, 255).astype(np.uint8)

# Save the combined output
cv2.imwrite('output_combined.png', output_combined)  # Save the combined image

# Display the combined image in VS Code
cv2.imshow('Combined Output', output_combined)  # Show the image
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the display window

# Evaluation Metrics:

# 1. Mean Squared Error (MSE)
mse_value = np.mean((ground_truth - output_combined) ** 2)

# 2. Peak Signal-to-Noise Ratio (PSNR)
if mse_value == 0:
    psnr_value = 100  # Infinite PSNR if MSE is 0 (perfect match)
else:
    psnr_value = 10 * math.log10((255 ** 2) / mse_value)

# 3. Structural Similarity Index (SSIM)
ssim_value, _ = ssim(ground_truth, output_combined, full=True, multichannel=True, win_size=3)

# Print the evaluation metrics
print(f'Mean Squared Error (MSE): {mse_value}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB')
print(f'Structural Similarity Index (SSIM): {ssim_value}')
