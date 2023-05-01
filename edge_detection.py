import cv2
import numpy as np
from skimage.util import random_noise

import color_spaces

# 2 EDGE DETECTION

# 2.1
# Loading the images
beach_image = cv2.imread('Utils/beach.jpg', cv2.IMREAD_GRAYSCALE)
diag2_image = cv2.imread('Utils/diag2.jpg', cv2.IMREAD_GRAYSCALE)

# Rescaling images to appropriate sizes
beach_image = color_spaces.scale_image(0.2, beach_image)
diag2_image = color_spaces.scale_image(0.13, diag2_image)

# Creating 45◦ and 135◦ 1st order spatial edge detection filters
filter_45 = np.array([[2,  1,  0],
                      [1,  0, -1],
                      [0, -1, -2]])

filter_135 = np.array([[0,  1, 2],
                      [-1,  0, 1],
                      [-2, -1, 0]])


# Applies the filter2D inbuilt function to the image, sets the pixel values to be unsigned 8-bit integers,
# it displays the raw filtered image, binarizes the filtered image and displays it.
def display_raw_and_binarized_filtered_image(image, filter, bin_threshold:int, degree:int, noise:str):
    # Applying the filter2D inbuilt function to the image
    filtered_image = cv2.filter2D(image, cv2.CV_16S, filter)

    # Calculating the absolute values and rescaling them to be in the range of 0 to 255
    abs_filtered_image = cv2.normalize(np.abs(filtered_image), None, 0, 255, cv2.NORM_MINMAX)

    # Converting the 16-bit signed integer type image to 8-bit unsigned integer type for display
    filtered_image = cv2.convertScaleAbs(abs_filtered_image)

    # Displaying the raw filtered image (before the step to binarize the filtered image)
    color_spaces.display_image('Raw ' + str(degree) + ' Degree ' + noise + 'Filtered Beach Image', filtered_image)

    # Binarize the filtering result
    _, binarized_image = cv2.threshold(filtered_image, bin_threshold, 255, cv2.THRESH_BINARY)

    # Displaying the binarized result
    color_spaces.display_image('Binarized ' + str(degree) + ' Degree ' + noise + 'Filtered Beach Image', binarized_image)


# Applies a de-noising filter
def median_filter(image, size):
    output = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            values = []
            for i in range(int(max(0, x-(size-1)/2)), int(min(image.shape[0]-1, x + (size-1)/2))):
                for j in range(int(max(0, y-(size-1)/2)), int(min(image.shape[1]-1, y + (size-1)/2))):
                    values.append(image[i][j])
            output[x][y] = np.median(values)
    return output


# Applying filtering to the images using the filter2d inbuilt function and binarizing the filtering results
display_raw_and_binarized_filtered_image(beach_image, filter_45, 130, 45, '')
display_raw_and_binarized_filtered_image(beach_image, filter_135, 130, 135, '')

display_raw_and_binarized_filtered_image(diag2_image, filter_45, 200, 45, '')
display_raw_and_binarized_filtered_image(diag2_image, filter_135, 180, 135, '')


# 2.2

# Adding salt and pepper noise to the images
sp_beach_image = random_noise(beach_image, mode='s&p', seed=0)
sp_beach_image = (255*sp_beach_image).astype(np.uint8)

sp_diag2_image = random_noise(diag2_image, mode='s&p', seed=0)
sp_diag2_image = (255*sp_diag2_image).astype(np.uint8)

# Applying filtering to the images (with salt and pepper noise) using the filter2d inbuilt function and binarizing the filtering results
display_raw_and_binarized_filtered_image(sp_beach_image, filter_45, 130, 45, 'Salt and Pepper Noise ')
display_raw_and_binarized_filtered_image(sp_beach_image, filter_135, 130, 135, 'Salt and Pepper Noise ')

display_raw_and_binarized_filtered_image(sp_diag2_image, filter_45, 200, 45, 'Salt and Pepper Noise ')
display_raw_and_binarized_filtered_image(sp_diag2_image, filter_135, 160, 135, 'Salt and Pepper Noise ')

# Applying a de-noising filter to the images
size = 3
denoised_beach_image = median_filter(sp_beach_image, size)
denoised_diag2_image = median_filter(sp_diag2_image, size)

# Applying filtering to the images (after applying the de-noising filter) using the filter2d inbuilt function and binarizing the filtering results
display_raw_and_binarized_filtered_image(denoised_beach_image, filter_45, 130, 45, 'S$P De-noise ')
display_raw_and_binarized_filtered_image(denoised_beach_image, filter_135, 130, 135, 'S$P De-noise ')

display_raw_and_binarized_filtered_image(denoised_diag2_image, filter_45, 200, 45, 'S$P De-noise ')
display_raw_and_binarized_filtered_image(denoised_diag2_image, filter_135, 160, 135, 'S$P De-noise ')


# 2.3

# Add Gaussian noise to the images
variance = 100
standard_deviation = np.sqrt(variance)

gaussian_noise_for_beach_image = np.random.normal(0, standard_deviation, beach_image.shape)
gaussian_noisy_beach_image = np.clip(beach_image + gaussian_noise_for_beach_image, 0, 255).astype(np.uint8)

gaussian_noise_diag2_image = np.random.normal(0, standard_deviation, diag2_image.shape)
gaussian_noisy_diag2_image = np.clip(diag2_image + gaussian_noise_diag2_image, 0, 255).astype(np.uint8)

# Applying filtering to the images (with Gaussian noise) using the filter2d inbuilt function and binarizing the filtering results
display_raw_and_binarized_filtered_image(gaussian_noisy_beach_image, filter_45, 130, 45, 'Gaussian Noise ')
display_raw_and_binarized_filtered_image(gaussian_noisy_beach_image, filter_135, 130, 135, 'Gaussian Noise ')

display_raw_and_binarized_filtered_image(gaussian_noisy_diag2_image, filter_45, 200, 45, 'Gaussian Noise ')
display_raw_and_binarized_filtered_image(gaussian_noisy_diag2_image, filter_135, 160, 135, 'Gaussian Noise ')

# Applying a de-noising filter (Gaussian filter) to the images
denoised_gaussian_noisy_beach_image = cv2.GaussianBlur(gaussian_noisy_beach_image, (3, 3), 0)
denoised_gaussian_noisy_diag2_image = cv2.GaussianBlur(gaussian_noisy_diag2_image, (3, 3), 0)

# Applying filtering to the images (Gaussian denoised) using the filter2d inbuilt function and binarizing the filtering results
display_raw_and_binarized_filtered_image(denoised_gaussian_noisy_beach_image, filter_45, 130, 45, 'Gaussian De-noise ')
display_raw_and_binarized_filtered_image(denoised_gaussian_noisy_beach_image, filter_135, 130, 135, 'Gaussian De-noise ')

display_raw_and_binarized_filtered_image(denoised_gaussian_noisy_diag2_image, filter_45, 200, 45, 'Gaussian De-noise ')
display_raw_and_binarized_filtered_image(denoised_gaussian_noisy_diag2_image, filter_135, 160, 135, 'Gaussian De-noise ')
