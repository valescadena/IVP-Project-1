import cv2
import numpy as np
import matplotlib.pyplot as plt

import color_spaces


# Loading the image
birdie_image = cv2.imread('Utils/birdie.jpg', cv2.IMREAD_GRAYSCALE)

width = birdie_image.shape[1]
height = birdie_image.shape[0]

x_row = np.arange(int(-width / 2), int(width / 2))
x_column = np.arange(int(-height / 2), int(height / 2))
axis_shift = (int(-width / 2), int(width / 2), int(-height / 2), int(height / 2))


# Computes the FFT magnitude centered in 2D
def compute_fft_2D_magnitude_centered(image):
    # Finding its 2D FFT
    fft_2D = np.fft.fft2(image)

    # Shifting the zero-frequency component to the center
    fft_2D_centered = np.fft.fftshift(fft_2D)

    # Returning the magnitude of the centered 2D FFT
    return fft_2D_centered


# Displays a 2D FFT and the 1D slice of the FFT magnitude from the middle row and column
def display_2D_and_1D_FFT(title:str, title_1D:str, image):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    image1 = ax1.imshow(np.log(image + 1e-4), extent=axis_shift)
    ax1.set_title(title)
    plt.colorbar(image1, shrink=0.4)
    image2 = ax2.imshow(np.log(image + 1e-4), extent=axis_shift)
    ax2.set_title('Zoomed in')
    ax2.set_xlim((-30, 30))
    ax2.set_ylim((-30, 30))
    plt.colorbar(image2, shrink=0.5)
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    image[image < 1e-8] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(x_row, image[int(height / 2), :])
    ax1.set_xlim((-20, 20))
    ax1.set_title('1D Middle Row Slice' + title_1D)
    ax2.plot(x_column, image[:, int(width / 2)])
    ax2.set_xlim((-20, 20))
    ax2.set_title('1D Middle Column Slice' + title_1D)
    plt.show()


# 3 FOURIER TRANSFORM

# 3.1

# Creating x and y meshgrid
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x_mesh_grid, y_mesh_grid = np.meshgrid(x, y)

# Creating a 2D sine wave with amplitude and frequency of choosing
amplitude = 50
frequency = 8
sine_wave_2D = amplitude * np.sin(frequency * 2 * np.pi * x_mesh_grid / width)

# Add 128 so the negative values can be positive (so not all negative values are shown as black pixels)
sine_wave_2D_disp = (128 + sine_wave_2D).astype(np.int16)
sine_wave_2D_disp = np.clip(sine_wave_2D_disp, 0, 255)
sine_wave_2D_disp = sine_wave_2D_disp.astype(np.uint8)

# Displaying the 2D sine of same size as the birdie_image
color_spaces.display_image('2D Sine Wave', color_spaces.scale_image(0.2, sine_wave_2D_disp))


# 3.2

# Finding the 2D FFT centered in 2D
fft_2D_centered = compute_fft_2D_magnitude_centered(sine_wave_2D)

# Displaying the 2D FFT magnitude centered in 2D and the 1D slice of the FFT magnitude from the middle row and column
display_2D_and_1D_FFT('FFT Magnitude Centered in 2D of Sine', '', np.abs(fft_2D_centered))


# 3.3


# Executes all steps in 3.3: creates and displays birdie image after adding the 2D sine wave. Calculates
# its FFT and displays FFT's magnitude centered in 2D. Displays 1D slice of FFT magnitude from the middle row and column
# Returns the FFT centered in 2D
def display_noisy_image_with_1D_and_2D_FFT(sine_wave_2D, title_amp:str):
    # Creating an image corrupted by periodic noise by adding the 2D sine to birdie_image
    noise_in_birdie_image = sine_wave_2D + birdie_image
    noise_in_birdie_image = cv2.convertScaleAbs(np.where(noise_in_birdie_image < 0, 0, noise_in_birdie_image))
    noise_in_birdie_image_disp = color_spaces.scale_image(0.4, noise_in_birdie_image)

    # Displaying the resulting image
    color_spaces.display_image('Birdie Image Corrupted by Periodic Noise' + title_amp, noise_in_birdie_image_disp)

    # Calculating the FFT of the resulting noisy image and displaying its magnitude centered in 2D
    fft_2D_centered_noise_birdie = compute_fft_2D_magnitude_centered(noise_in_birdie_image)

    # Displaying its magnitude centered in 2D and the 1D slice of the FFT magnitude from the middle row and column
    display_2D_and_1D_FFT('FFT Magnitude Centered after Noise', ' After Noise', np.abs(fft_2D_centered_noise_birdie))
    return fft_2D_centered_noise_birdie


fft_2D_centered_noise_birdie = display_noisy_image_with_1D_and_2D_FFT(sine_wave_2D, ' for Amplitude of ' + str(amplitude))

# 3.4


# Executes all steps in 3.4:
def denoising_filter_and_display_denoised_image(fft_2D_centered_noise_birdie, title_amp:str):
        # Creating a frequency domain filter to remove the periodic noise
    frequency_filter = np.ones((height, width))
    frequency_filter[int(height / 2)][int(width / 2) + frequency] = 0
    frequency_filter[int(height / 2)][int(width / 2) - frequency] = 0


    # (1) Displaying the de-noising filter’s FT magnitude in 2D and 1D
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(frequency_filter, cmap='gray', extent=axis_shift)
    ax1.set_title('De-noising filter’s FT magnitude in 2D')
    ax2.imshow(frequency_filter, cmap='gray', extent=axis_shift)
    ax2.set_title('Zoomed in')
    ax2.set_xlim((-30, 30))
    ax2.set_ylim((-30, 30))
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(x_row, frequency_filter[int(height / 2), :])
    ax1.set_xlim((-20, 20))
    ax1.set_title('1D Middle Row Slice of De-noising Filter')
    ax2.plot(x_column, frequency_filter[:, int(width / 2)])
    ax2.set_xlim((-20, 20))
    ax2.set_title('1D Middle Column Slice of De-noising Filter')
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    # (2) Displaying the de-noised image’s FT magnitude in 1D and 2D
    denoised_fft_2D = fft_2D_centered_noise_birdie * frequency_filter

    display_2D_and_1D_FFT('De-noised FFT', ' After De-noising', np.abs(denoised_fft_2D))

    # (3) Displaying the resulting de-noised image
    denoised_birdie_image = np.fft.ifft2(np.fft.ifftshift(denoised_fft_2D)).astype(np.uint8)

    color_spaces.display_image('De-noised Birdie Image' + title_amp, color_spaces.scale_image(0.4, denoised_birdie_image))


denoising_filter_and_display_denoised_image(fft_2D_centered_noise_birdie, ' for Amplitude of ' + str(amplitude))

# 3.5

# Creating sine wave with same frequency but different amplitude
amplitude = 100
sine_wave_2D = amplitude * np.sin(frequency * 2 * np.pi * x_mesh_grid / width)

# Repeating the steps in 3.3
fft_2D_centered_noise_birdie = display_noisy_image_with_1D_and_2D_FFT(sine_wave_2D, ' for Amplitude of ' + str(amplitude))

# Repeating the steps in 3.4
denoising_filter_and_display_denoised_image(fft_2D_centered_noise_birdie, ' for Amplitude of ' + str(amplitude))
