import cv2
import os
import glob

def resize_images(input_dir, output_dir, size=(64, 64)):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all image file paths in the input directory
    image_paths = glob.glob(os.path.join(input_dir, '*'))

    for img_path in image_paths:
        # Read the image
        img = cv2.imread(img_path)

        # Resize the image
        resized_img = cv2.resize(img, size)

        # Extract filename and create output path
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)

        # Save the resized image
        cv2.imwrite(output_path, resized_img)
        print(f"Resized and saved {filename}")

# Example usage
input_dir = r'C:\Users\aishw\Documents\zklabs\CVAT_Patches\two_wheelers'  # Replace with your input directory
output_dir = input_dir + '_resized_64'  # Output directory
resize_images(input_dir, output_dir)