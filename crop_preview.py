from PIL import Image
import matplotlib.pyplot as plt

# List of 5 sample spectrogram image paths
img_paths = [
    "artifacts/run-uyaqujc2-spectrograms:v0/media/images/ffd2fc476cea050e4d85/spectrogram_667.png",
    "artifacts/run-uyaqujc2-spectrograms:v0/media/images/ffdc33eb06815da37bdd/spectrogram_3644.png",
    "artifacts/run-uyaqujc2-spectrograms:v0/media/images/ffe022951339ce7a280f/spectrogram_2736.png",
    "artifacts/run-uyaqujc2-spectrograms:v0/media/images/fff448f98c07ddc5d484/spectrogram_2842.png",
    "artifacts/run-uyaqujc2-spectrograms:v0/media/images/fff452877d6a4356f3c5/spectrogram_4571.png",
]

# Cropping dimensions (left, upper, right, lower)
crop_box = (80, 60, 780, 340)  # Crops out even more from the right to ensure colorbar is removed

plt.figure(figsize=(15, 15))
for i, path in enumerate(img_paths):
    img = Image.open(path)
    print(f"Image {i+1} width: {img.width} height: {img.height}")
    # Calculate right and lower based on image size if negative
    left, upper, right, lower = crop_box
    print(f"Cropping: left={left}, upper={upper}, right={right}, lower={lower}")
    img_cropped = img.crop((left, upper, right, lower))
    
    # Show original, cropped, and rightmost 100px of original
    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Original {i+1}')
    
    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(img_cropped)
    plt.axis('off')
    plt.title(f'Cropped {i+1}')

    # Rightmost 200px of original
    rightmost = img.crop((img.width - 200, 0, img.width, img.height))
    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(rightmost)
    plt.axis('off')
    plt.title(f'Rightmost 200px {i+1}')
plt.tight_layout()
plt.savefig('original_vs_cropped_vs_rightmost.png')
plt.show() 