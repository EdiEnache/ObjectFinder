import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageTk

from helper import preprocess_image, segment_image, threshold_values, remove_small_regions, color_regions, \
    colorize_regions, compare_regions, calculate_shape_parameters

imported_image1 = None
preprocessed_image1 = None
objects_image1 = None
current_threshold1 = []
saved_image1 = None
colored_image1 = None
region_parameters1 = []

imported_image2 = None
preprocessed_image2 = None
objects_image2 = None
current_threshold2 = []
saved_image2 = None
colored_image2 = None
region_parameters2 = []

def import_image_left():
    global imported_image1
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")])
    if file_path:
        imported_image1 = Image.open(file_path)
        imported_image1.thumbnail((800, 800))  # Resize the image to fit within the frame
        photo = ImageTk.PhotoImage(imported_image1)
        image_label.configure(image=photo)  # Update the image label with the new image
        image_label.image = photo  # Keep a reference to the photo to prevent it from being garbage collected

def import_image_right():
    global imported_image2
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")])
    if file_path:
        imported_image2 = Image.open(file_path)
        imported_image2.thumbnail((800, 800))  # Resize the image to fit within the frame
        photo = ImageTk.PhotoImage(imported_image2)
        right_image_label.configure(image=photo)  # Update the image label in the right frame
        right_image_label.image = photo  # Keep a reference to the photo to prevent it from being garbage collected



def preprocess1():
    global imported_image1, preprocessed_image1

    if imported_image1:
        preprocessed_image1 = preprocess_image(imported_image1)
        photo = ImageTk.PhotoImage(preprocessed_image1)
        image_label.configure(image=photo)  # Update the image label with the new image
        image_label.image = photo  # Keep a reference to the photo to prevent it from being garbage collected
    else:
        messagebox.showwarning("No Image", "Please import an image for processing.")

def preprocess2():
    global imported_image2, preprocessed_image2

    if imported_image2:
        preprocessed_image2 = preprocess_image(imported_image2)
        photo = ImageTk.PhotoImage(preprocessed_image2)
        right_image_label.configure(image=photo)  # Update the image label in the right frame
        right_image_label.image = photo  # Keep a reference to the photo to prevent it from being garbage collected
    else:
        messagebox.showwarning("No Image", "Please import an image for processing.")


def objects1():
    global preprocessed_image1, objects_image1

    if preprocessed_image1:
        objects_image1 = segment_image(preprocessed_image1)
        photo = ImageTk.PhotoImage(objects_image1)
        image_label.configure(image=photo)  # Update the image label with the new image
        image_label.image = photo  # Keep a reference to the photo to prevent it from being garbage collected
        remove_small_objects1()
    else:
        messagebox.showwarning("No Preprocessed Image", "Please preprocess an image before segmenting objects.")

def objects2():
    global preprocessed_image2, objects_image2

    if preprocessed_image2:
        objects_image2 = segment_image(preprocessed_image2)
        photo = ImageTk.PhotoImage(objects_image2)
        right_image_label.configure(image=photo)  # Update the image label with the new image
        right_image_label.image = photo  # Keep a reference to the photo to prevent it from being garbage collected
        remove_small_objects2()
    else:
        messagebox.showwarning("No Preprocessed Image", "Please preprocess an image before segmenting objects.")

def remove_small_objects1():
    global objects_image1, current_threshold1
    if not objects_image1:
        messagebox.showwarning("Segmentation Error", "Please segment the image first.")
        return

    # Create the popup window
    popup = tk.Toplevel(window)
    popup.title("Small Objects Threshold")

    # Create the threshold slider
    slider_label = tk.Label(popup, text="Threshold")
    slider_label.pack()

    largest_region_size, smallest_region_size = threshold_values(objects_image1)
    current_threshold1.append(smallest_region_size)
    threshold_slider = tk.Scale(popup, from_=smallest_region_size, to=largest_region_size, orient=tk.HORIZONTAL, length=200)
    threshold_slider.set(smallest_region_size)
    threshold_slider.pack()

    # Create the Apply button
    def apply_threshold():
        current_threshold1[0] = threshold_slider.get()
        photo = remove_small_regions(objects_image1, current_threshold1[0])
        image_label.configure(image=photo)
        image_label.image = photo

    apply_button = tk.Button(popup, text="Apply", command=apply_threshold)
    apply_button.pack()

    def save_threshold():
        global saved_image1

        saved_image1 = remove_small_regions(objects_image1, current_threshold1[0])
        saved_image1 = ImageTk.getimage(saved_image1)
        saved_image1 = np.array(saved_image1)  # Convert to NumPy array
        saved_image1 = cv2.cvtColor(saved_image1, cv2.COLOR_BGR2GRAY)
        print(type(saved_image1))
        popup.destroy()  # Close the popup window

    save_button = tk.Button(popup, text="Save", command=save_threshold)
    save_button.pack()



def remove_small_objects2():
    global objects_image2, current_threshold2
    if not objects_image2:
        messagebox.showwarning("Segmentation Error", "Please segment the image first.")
        return

    # Create the popup window
    popup = tk.Toplevel(window)
    popup.title("Small Objects Threshold")

    # Create the threshold slider
    slider_label = tk.Label(popup, text="Threshold")
    slider_label.pack()

    largest_region_size, smallest_region_size = threshold_values(objects_image2)
    current_threshold2 = [smallest_region_size]  # Initialize with the smallest region size
    threshold_slider = tk.Scale(popup, from_=smallest_region_size, to=largest_region_size, orient=tk.HORIZONTAL, length=200)
    threshold_slider.set(smallest_region_size)
    threshold_slider.pack()

    # Create the Apply button
    def apply_threshold():
        current_threshold2[0] = threshold_slider.get()
        photo = remove_small_regions(objects_image2, current_threshold2[0])
        right_image_label.configure(image=photo)
        right_image_label.image = photo

    apply_button = tk.Button(popup, text="Apply", command=apply_threshold)
    apply_button.pack()

    def save_threshold():
        global saved_image2

        saved_image2 = remove_small_regions(objects_image2, current_threshold2[0])
        saved_image2 = ImageTk.getimage(saved_image2)
        saved_image2 = np.array(saved_image2)  # Convert to NumPy array
        saved_image2 = cv2.cvtColor(saved_image2, cv2.COLOR_BGR2GRAY)
        popup.destroy()  # Close the popup window

    save_button = tk.Button(popup, text="Save", command=save_threshold)
    save_button.pack()

def color1():
    global saved_image1, region_parameters1

    if saved_image1 is None or saved_image1.size == 0:
        messagebox.showwarning("No Image", "Please color the image first.")
        return

    # Call the color_regions function
    colored_image1, region_parameters1 = color_regions(saved_image1)

    # Convert the colored image to a NumPy array
    colored_array = np.array(colored_image1)

    # Convert the NumPy array to a PIL Image
    colored_pil = Image.fromarray(colored_array)

    # Convert the PIL Image to a PhotoImage
    photo = ImageTk.PhotoImage(colored_pil)

    image_label.configure(image=photo)
    image_label.image = photo

    # Create a Toplevel window for displaying shape parameters and colors
    window = tk.Toplevel()
    window.title('Shape Parameters and Colors (Left)')

    # Create a message frame
    message_frame = tk.Frame(window)
    message_frame.pack(padx=10, pady=10)

    # Display shape parameters and color swatches in the message frame
    for i, region in enumerate(region_parameters1):
        # Get the color information
        color = region['Color']

        # Create a label with the color swatch
        color_label = tk.Label(message_frame, bg='#%02x%02x%02x' % color, width=10, height=2)
        color_label.grid(row=i, column=0, padx=5, pady=5)

        # Add the shape parameters to the message
        for j, (key, value) in enumerate(region.items()):
            if key != 'Color':
                parameter_label = tk.Label(message_frame, text=f'{key}: {value}')
                parameter_label.grid(row=i, column=j+1, padx=5, pady=5)

    # Run the main Tkinter event loop for the Toplevel window
    window.mainloop()


def recognition():
    global saved_image2, region_parameters1, region_parameters2

    if saved_image2 is None or saved_image2.size == 0:
        messagebox.showwarning("No Image", "Please load the image first.")
        return

    # Calculate region parameters for saved_image2
    np_segmented_image2 = np.array(saved_image2)
    num_labels, labels = cv2.connectedComponents(np_segmented_image2)
    region_parameters2 = []

    for label in range(1, np.max(labels) + 1):
        region_mask = (labels == label).astype(np.uint8)
        parameters = calculate_shape_parameters(region_mask)

        region_info = {
            'Label': label,
            'Area': parameters['Area'],
            'Perimeter': parameters['Perimeter'],
            'Compactness': parameters['Compactness'],
            'Aspect Ratio': parameters['Aspect Ratio'],
            'Solidity': parameters['Solidity'],
            'Euler Number': parameters['Euler Number'],
            'Color': None  # Placeholder for assigned color
        }






        region_parameters2.append(region_info)

        # Compare region parameters with region_parameters1
        for region2 in region_parameters2:
            for region1 in region_parameters1:
                if compare_regions(region1, region2):
                    region2['Color'] = region1['Color']
                    break  # Exit the inner loop once a match is found


    print("REGION_PARAMETERS1:")
    # Print the region_parameters1
    for region in region_parameters1:
        print(region)

    print("REGION_PARAMETERS2:")
    # Print the region_parameters2
    for region in region_parameters2:
        print(region)

        # Perform connected component labeling
    _, labels = cv2.connectedComponents(saved_image2)

    # Convert labels to 8-bit unsigned integer type
    labels = np.uint8(labels)

    # Colorize the regions in saved_image2
    colored_image = colorize_regions(labels, region_parameters2)
    print(type(saved_image2))
    plt.imshow(saved_image2)
    plt.show()
    # Convert the colored image to PIL Image
    colored_pil = Image.fromarray(colored_image, 'RGB')

    # Convert the PIL Image to a PhotoImage
    photo = ImageTk.PhotoImage(colored_pil)

    # Display the colored image
    right_image_label.configure(image=photo)
    right_image_label.image = photo






# Main Window
window = tk.Tk()
window.geometry("1200x700")

# Left side frame
left_frame = tk.Frame(window, width=500, height=500)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

# Right side frame
right_frame = tk.Frame(window, width=500, height=500)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

# Toolbar for the left side
left_toolbar = tk.Frame(left_frame, height=30)
left_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# Button 1 for importing an image to the left frame
button1 = tk.Button(left_toolbar, text="Import Image", command=import_image_left)
button1.pack(side=tk.LEFT)

# Button 2 for preprocessing an image to the left frame
button2 = tk.Button(left_toolbar, text="Preprocess", command=preprocess1)
button2.pack(side=tk.LEFT)

# Button 3 for object recognition an image to the left frame
button3 = tk.Button(left_toolbar, text="Objects", command=objects1)
button3.pack(side=tk.LEFT)
# Button 3 for object recognition an image to the left frame
button3 = tk.Button(left_toolbar, text="Highlight", command=color1)
button3.pack(side=tk.LEFT)

# Toolbar for the right side
right_toolbar = tk.Frame(right_frame, height=30)
right_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# Button 1 for importing an image to the right frame
button1 = tk.Button(right_toolbar, text="Import Image", command=import_image_right)
button1.pack(side=tk.LEFT)
# Button 2 for preprocessing an image to the right frame
button1 = tk.Button(right_toolbar, text="Preprocess", command=preprocess2)
button1.pack(side=tk.LEFT)
# Button 3 for object recognition an image to the right frame
button1 = tk.Button(right_toolbar, text="Objects", command=objects2)
button1.pack(side=tk.LEFT)
# Button 3 for object recognition an image to the right frame
button1 = tk.Button(right_toolbar, text="Recognition", command=recognition)
button1.pack(side=tk.LEFT)



# Image label to display the imported image in the left frame
image_label = tk.Label(left_frame)  # Create a label widget
image_label.pack()  # Pack the label into the left frame

# Image label to display the imported image in the right frame
right_image_label = tk.Label(right_frame)  # Create a label widget
right_image_label.pack()  # Pack the label into the right frame

# Start the main loop
window.mainloop()
