import cv2
import numpy as np
import matplotlib.pyplot as plt


def scale_image(scalar, image):
    return cv2.resize(image, (int(image.shape[1]*scalar), int(image.shape[0] * scalar)))

# To display images in a grid (a big image containing images next to each other)
def display_images_in_grid(title:str, num_rows:int, num_columns:int, images:tuple):
    grid_img_width = 0
    grid_img_height = 0
    for i in range(num_rows):
        temp_img_width = 0
        max_img_height = 0
        for j in range(num_columns):
            index_image = i * num_columns + j
            temp_img_width = temp_img_width + images[index_image].shape[1]
            if max_img_height < images[index_image].shape[0]:
                max_img_height = images[index_image].shape[0]
        grid_img_height = grid_img_height + max_img_height
        if grid_img_width < temp_img_width:
            grid_img_width = temp_img_width

    # Create a new image to display the images side by side
    grid_img = np.zeros((grid_img_height, grid_img_width), np.uint8)

    x = 0
    y = 0
    y_max_temp = 0
    # Copy each image to the corresponding region in the grid image
    for i in range(num_rows):
        for j in range(num_columns):
            index_image = i * num_columns + j
            grid_img[y:y + images[index_image].shape[0], x:x + images[index_image].shape[1]] = images[index_image]
            x = x + images[index_image].shape[1]
            if y_max_temp < images[index_image].shape[0]:
                y_max_temp = images[index_image].shape[0]
            if j == num_columns-1:
                x = 0
                y = y + y_max_temp
                y_max_temp = 0
    display_image(title, grid_img)


def display_image(title:str, image):
    # Display the grid image
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1 COLOR SPACES
if __name__ == "__main__":
    # 1.1
    # My high contrast image is the flamingo image and my low contrast image is the deer image
    # Loading the high and low contrast images
    flamingo_image = cv2.imread('Utils/flamingo.jpg')
    deer_image = cv2.imread('Utils/deer.jpg')

    # Scaling the images to an appropriate size
    flamingo_image = scale_image(0.1, flamingo_image)
    deer_image = scale_image(0.2, deer_image)

    # Converting the RGB images to HSV
    hsv_flamingo_image = cv2.cvtColor(flamingo_image, cv2.COLOR_BGR2HSV)
    hsv_deer_image = cv2.cvtColor(deer_image, cv2.COLOR_BGR2HSV)

    flamingo_grid_scalar = 0.5
    deer_grid_scalar = 0.48

    # Scale the images by the respective scalar to fit in grid
    flamingo_grid_size_image = scale_image(flamingo_grid_scalar, flamingo_image)
    deer_grid_size_image = scale_image(deer_grid_scalar, deer_image)

    hsv_flamingo_grid_size_image = scale_image(flamingo_grid_scalar, hsv_flamingo_image)
    hsv_deer_grid_size_image = scale_image(deer_grid_scalar, hsv_deer_image)

    # Getting the BGR values of the flamingo and deer image
    b_flamingo_values, g_flamingo_values, r_flamingo_values = cv2.split(flamingo_grid_size_image)
    b_deer_values, g_deer_values, r_deer_values = cv2.split(deer_grid_size_image)

    # Getting the HSV values of the flamingo and deer image
    h_flamingo_values, s_flamingo_values, v_flamingo_values = cv2.split(hsv_flamingo_grid_size_image)
    h_deer_values, s_deer_values, v_deer_values = cv2.split(hsv_deer_grid_size_image)

    # display_images_in_grid('HSV Flamingo and Deer Images', 1, 2, (hsv_flamingo_image, hsv_deer_image))
    display_image('HSV Flamingo Image', hsv_flamingo_image)
    display_image('HSV Deer Image', hsv_deer_image)

    # Displaying each color channel (R, G, B and H, S, V)
    display_images_in_grid('B G R Flamingo and Deer Images', 2, 3, (b_flamingo_values, g_flamingo_values, r_flamingo_values, b_deer_values, g_deer_values, r_deer_values))
    display_images_in_grid('H S V Flamingo and Deer Images', 2, 3, (h_flamingo_values, s_flamingo_values, v_flamingo_values, h_deer_values, s_deer_values, v_deer_values))


    # 1.2
    # Applying histogram equalization to the H, S, V channels separately from the Flamingo image
    h_equalized_flamingo_values = cv2.equalizeHist(h_flamingo_values)
    s_equalized_flamingo_values = cv2.equalizeHist(s_flamingo_values)
    v_equalized_flamingo_values = cv2.equalizeHist(v_flamingo_values)

    # Applying histogram equalization to the H, S, V channels separately from the Deer image
    h_equalized_deer_values = cv2.equalizeHist(h_deer_values)
    s_equalized_deer_values = cv2.equalizeHist(s_deer_values)
    v_equalized_deer_values = cv2.equalizeHist(v_deer_values)

    # Resulting HSV color images
    hsv_equalized_flamingo_values = scale_image(1.0 / flamingo_grid_scalar, cv2.merge((h_equalized_flamingo_values, s_equalized_flamingo_values, v_equalized_flamingo_values)))
    hsv_equalized_deer_values = scale_image(1.0 / deer_grid_scalar, cv2.merge((h_equalized_deer_values, s_equalized_deer_values, v_equalized_deer_values)))

    # Displaying the histogram equalized channels for both images
    display_images_in_grid('H S V Separate (Equalized) Flamingo and Deer Images Histogram', 2, 3, (h_equalized_flamingo_values, s_equalized_flamingo_values, v_equalized_flamingo_values, h_equalized_deer_values, s_equalized_deer_values, v_equalized_deer_values))

    # Displaying the resulting color images
    display_image('Resulting HSV Equalization Flamingo Image', hsv_equalized_flamingo_values)
    display_image('Resulting HSV Equalization Deer Image', hsv_equalized_deer_values)

    # Plotting the resulting histograms
    plt.subplot(231)
    plt.hist(h_equalized_flamingo_values.ravel(), 256, [0, 256])
    plt.title('H Eq. Flamingo Img')
    plt.subplot(232)
    plt.hist(s_equalized_flamingo_values.ravel(), 256, [0, 256])
    plt.title('S Eq. Flamingo Img')
    plt.subplot(233)
    plt.hist(v_equalized_flamingo_values.ravel(), 256, [0, 256])
    plt.title('V Eq. Flamingo Img')
    plt.subplots_adjust(wspace=0.9)
    plt.show()

    plt.subplot(231)
    plt.hist(h_equalized_deer_values.ravel(), 256, [0, 256])
    plt.title('H Eq. Deer Image')
    plt.subplot(232)
    plt.hist(s_equalized_deer_values.ravel(), 256, [0, 256])
    plt.title('S Eq. Deer Image')
    plt.subplot(233)
    plt.hist(v_equalized_deer_values.ravel(), 256, [0, 256])
    plt.title('V Eq. Deer Image')
    plt.subplots_adjust(wspace=0.5)
    plt.show()


    # 1.3
    # Displaying the histogram equalized channels for both images
    display_images_in_grid('H S V Separate (V Equalized) Flamingo and Deer Images Histogram', 2, 3, (scale_image(2, h_flamingo_values), scale_image(2, s_flamingo_values), scale_image(2, v_equalized_flamingo_values), scale_image(2, h_deer_values), scale_image(2, s_deer_values), scale_image(2, v_equalized_deer_values)))

    # Plotting the resulting histograms
    plt.subplot(231)
    plt.hist(h_flamingo_values.ravel(), 256, [0, 256])
    plt.title('H Flamingo Image')
    plt.subplot(232)
    plt.hist(s_flamingo_values.ravel(), 256, [0, 256])
    plt.title('S Flamingo Image')
    plt.subplot(233)
    plt.hist(v_equalized_flamingo_values.ravel(), 256, [0, 256])
    plt.title('V Eq. Flamingo Image')
    plt.subplots_adjust(wspace=0.5 )
    plt.show()

    plt.subplot(231)
    plt.hist(h_deer_values.ravel(), 256, [0, 256])
    plt.title('H Deer Image')
    plt.subplot(232)
    plt.hist(s_deer_values.ravel(), 256, [0, 256])
    plt.title('S Deer Image')
    plt.subplot(233)
    plt.hist(v_equalized_deer_values.ravel(), 256, [0, 256])
    plt.title('V Eq. Deer Image')
    plt.subplots_adjust(wspace=0.7)
    plt.show()

    # Resulting HSV color images
    hsv_v_equalized_flamingo_values = cv2.merge((h_flamingo_values, s_flamingo_values, v_equalized_flamingo_values))
    hsv_v_equalized_deer_values = cv2.merge((h_deer_values, s_deer_values, v_equalized_deer_values))

    # Displaying the resulting color images
    display_image('Resulting HSV Equalization Flamingo Image', scale_image(4, hsv_v_equalized_flamingo_values))
    display_image('Resulting HSV Equalization Deer Image', scale_image(4, hsv_v_equalized_deer_values))
