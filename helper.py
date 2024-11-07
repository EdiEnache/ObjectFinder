from tkinter import filedialog

import cv2
from PIL import Image, ImageTk, ImageOps
from scipy.ndimage import median_filter
import numpy as np
from skimage.filters import threshold_otsu


def preprocess_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Normalize the pixel values
    normalized_image = ImageOps.autocontrast(grayscale_image)

    # Apply a Gaussian filter to the normalized image
    filtered_image = median_filter(np.array(normalized_image), size=3)

    # Convert the filtered image back to PIL Image object
    filtered_image = Image.fromarray(filtered_image)

    # Return the preprocessed image
    return filtered_image


def segment_image(image):
    # Open the image using Pillow
    image = image

    # Convert the grayscale image to a NumPy array
    np_image = np.array(image)

    # Apply Otsu's thresholding
    threshold_value = threshold_otsu(np_image)
    thresholded_image = np_image > threshold_value

    # Convert the NumPy array back to a PIL Image
    segmented_image = Image.fromarray(np.uint8(thresholded_image) * 255)

    # Return the segmented image
    return segmented_image


def remove_small_regions(segmented_image, threshold):
    # Convert the segmented image to a NumPy array
    np_segmented_image = np.array(segmented_image)

    num_labels, labels = cv2.connectedComponents(np_segmented_image)

    region_sizes = np.bincount(labels.flatten())
    small_regions = np.where(region_sizes < threshold)[0]

    mask = np.isin(labels, small_regions)
    np_segmented_image[mask] = 0

    # Convert the NumPy array back to a PIL Image
    segmented_image = Image.fromarray(np_segmented_image)

    # Convert the PIL Image to PhotoImage
    photo = ImageTk.PhotoImage(segmented_image)

    return photo

def threshold_values(segmented_image):
    # Calculate the region sizes
    np_segmented_image = np.array(segmented_image)
    num_labels, labels = cv2.connectedComponents(np_segmented_image)

    region_sizes = np.bincount(labels.flatten())
    region_sizes = region_sizes[1:]  # Exclude the background region size

    # Find the size of the largest region
    largest_region_size = np.max(region_sizes)
    #Size of the smallest region
    smallest_region_size = np.min(region_sizes)



    # Calculate the recommended threshold deviation
    mean_size = np.mean(region_sizes)
    std_size = np.std(region_sizes)
    threshold_factor = 2.0  # Adjust this factor based on your requirements
    threshold_dev = threshold_factor * std_size

    return largest_region_size, smallest_region_size

def color_regions(saved_image):
    # Perform connected component labeling
    num_labels, labels = cv2.connectedComponents(saved_image)

    # Generate random colors for each region
    colors = []
    for _ in range(num_labels):
        colors.append((np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))

    # Create a blank colored image
    colored_image = np.zeros((*saved_image.shape, 3), dtype=np.uint8)

    # Create a list of dictionaries to store region parameters
    region_parameters = []

    # Color each region in the colored image and calculate shape parameters
    for label in range(1, num_labels):
        # Color the region in the colored image
        colored_image[labels == label] = colors[label]

        # Calculate shape parameters for the region
        region_mask = (labels == label).astype(np.uint8)
        parameters = calculate_shape_parameters(region_mask)

        # Calculate Euler number
        #connectivity_number = num_labels - 1
        #holes = parameters['Area'] - parameters['Perimeter'] + 1
        #euler_number = connectivity_number - holes

        # Create a dictionary for the region and store shape parameters and color
        region_info = {
            'Area': parameters['Area'],
            'Perimeter': parameters['Perimeter'],
            'Compactness': parameters['Compactness'],
            'Aspect Ratio': parameters['Aspect Ratio'],
            'Solidity': parameters['Solidity'],
            'Euler Number': parameters['Euler Number'],
            'Color': colors[label]
        }

        # Append the region dictionary to the region_parameters list
        region_parameters.append(region_info)

    # Convert the colored image to PIL Image
    colored_image = Image.fromarray(colored_image)

    # Return the colored image and region parameters
    return colored_image, region_parameters





def calculate_euler_number(region_mask):
    # Perform connected component labeling to identify objects and holes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)

    # Count the number of objects (excluding the background label)
    num_objects = num_labels - 1

    # Create a binary mask of the holes (label 0 represents the background)
    holes_mask = (labels > 0).astype(np.uint8)

    # Perform morphological operations to fill holes and identify connected components
    _, filled_mask = cv2.threshold(holes_mask, 0, 255, cv2.THRESH_BINARY_INV)
    num_holes, _, _, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)

    # Calculate the Euler number
    euler_number = num_objects - num_holes

    return euler_number



def calculate_shape_parameters(region_mask):
    # Calculate the contour of the region
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate area and perimeter
    area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)

    # Calculate compactness (skip if area is zero)
    compactness = 0.0 if area == 0 else perimeter ** 2 / (4 * np.pi * area)

    # Calculate the bounding rectangle
    x, y, w, h = cv2.boundingRect(contours[0])
    aspect_ratio = w / h

    # Calculate solidity
    convex_hull = cv2.convexHull(contours[0])
    convex_area = cv2.contourArea(convex_hull)

    if convex_area == 0:
        solidity = 0.0  # Set a default value or handle it based on your requirements
    else:
        solidity = area / convex_area

    # Calculate Euler number
    euler_number = calculate_euler_number(region_mask)

    # Create a dictionary to store the shape parameters
    parameters = {
        'Area': area,
        'Perimeter': perimeter,
        'Compactness': compactness,
        'Aspect Ratio': aspect_ratio,
        'Solidity': solidity,
        'Euler Number': euler_number
    }

    return parameters



def compare_regions(region1, region2):
    region1_parameters = {
        'Area': region1['Area'],
        'Perimeter': region1['Perimeter'],
        'Compactness': region1['Compactness'],
        'Aspect Ratio': region1['Aspect Ratio'],
        'Solidity': region1['Solidity'],
        'Euler Number': region1['Euler Number']
    }

    region2_parameters = {
        'Area': region2['Area'],
        'Perimeter': region2['Perimeter'],
        'Compactness': region2['Compactness'],
        'Aspect Ratio': region2['Aspect Ratio'],
        'Solidity': region2['Solidity'],
        'Euler Number': region2['Euler Number']
    }

    # Define the percentage threshold for error bounds
    error_threshold = 0.1/100  # Adjust this value as needed

    # Calculate the error bounds based on the parameter values
    error_bounds = {
        'Area': error_threshold * region1_parameters['Area'],
        'Perimeter': error_threshold * region1_parameters['Perimeter'],
        'Compactness': error_threshold * region1_parameters['Compactness'],
        'Aspect Ratio': error_threshold * region1_parameters['Aspect Ratio'],
        'Solidity': error_threshold * region1_parameters['Solidity'],
        'Euler Number': error_threshold * abs(region1_parameters['Euler Number'])
    }

    # Compare the region parameters within the calculated error bounds
    if (
        abs(region1_parameters['Area'] - region2_parameters['Area']) < error_bounds['Area'] and
        abs(region1_parameters['Perimeter'] - region2_parameters['Perimeter']) < error_bounds['Perimeter'] and
        abs(region1_parameters['Compactness'] - region2_parameters['Compactness']) < error_bounds['Compactness'] and
        abs(region1_parameters['Aspect Ratio'] - region2_parameters['Aspect Ratio']) < error_bounds['Aspect Ratio'] and
        abs(region1_parameters['Solidity'] - region2_parameters['Solidity']) < error_bounds['Solidity'] and
        abs(region1_parameters['Euler Number'] - region2_parameters['Euler Number']) < error_bounds['Euler Number']
    ):
        return True
    else:
        print(region2_parameters)
        return False



def colorize_regions(labels_image, region_parameters):
    colored_image = np.zeros((labels_image.shape[0], labels_image.shape[1], 3), dtype=np.uint8)

    for region in region_parameters:
        if 'Color' in region and region['Color'] is not None:
            label = region['Label']
            color = region['Color']

            colored_pixels = (labels_image == label)
            colored_indices = np.where(colored_pixels)
            colored_image[colored_indices] = color

    return colored_image






