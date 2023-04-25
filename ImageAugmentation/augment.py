import os
import random
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa

# Set the path to the directory containing the images
input_dir = '../../Documents/Cropped-Images/ScaledClusters/Cluster1-Training/'

# Set the path to the directory where the augmented images will be saved
output_dir = '../../Documents/Cropped-Images/ScaledClusters/Cluster1-Training-Augmented/'

# Define the augmentations to apply to the images
augmentations = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10)), # rotate by -10 to 10 degrees
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
])

# Get a list of all the image file names in the input directory
image_files = os.listdir(input_dir)

# Select 74 random image file names from the list
# random_files = random.sample(image_files, 74)
count = 0
for file_name in image_files:
    if count == 71:
        break
    # Load the image using PIL
    try:
        image = Image.open(os.path.join(input_dir, file_name)).convert('RGB')
    except:
        continue
    
    # Convert the PIL image to a numpy array
    image_array = np.array(image)
    
    # Apply the augmentations to the image
    augmented_image_array = augmentations(image=image_array)
    
    # Convert the augmented numpy array back to a PIL image
    augmented_image = Image.fromarray(np.uint8(augmented_image_array))
    
    # Save the augmented image to the output directory
    output_file_name = os.path.splitext(file_name)[0] + '_augmented.jpg'
    augmented_image.save(os.path.join(output_dir, output_file_name))

    count += 1
