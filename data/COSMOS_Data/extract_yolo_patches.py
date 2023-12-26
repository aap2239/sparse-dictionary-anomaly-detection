import cv2
import os
import glob

def extract_patches(image_path, annotation_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)

    with open(annotation_path, 'r') as file:
        for line in file:
            # Parse YOLO format: class_id, x_center, y_center, width, height
            class_id, x_center, y_center, width, height = map(float, line.split())

            # Convert YOLO coordinates to bounding box coordinates
            image_height, image_width = image.shape[:2]
            x_center, y_center, width, height = x_center * image_width, y_center * image_height, width * image_width, height * image_height
            x_min, y_min = int(x_center - width / 2), int(y_center - height / 2)
            x_max, y_max = int(x_center + width / 2), int(y_center + height / 2)

            # Extract the patch
            patch = image[y_min:y_max, x_min:x_max]

            # Create output directory for the class if it doesn't exist
            class_dir = os.path.join(output_dir, str(int(class_id)))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Save the patch
            patch_name = os.path.basename(image_path).split('.')[0] + f'_class_{int(class_id)}.jpg'
            cv2.imwrite(os.path.join(class_dir, patch_name), patch)

# Example usage
image_dir = r".\bike_annotations_20231120-141601\frames"
annotation_dir = r".\bike_annotations_20231120-141601\obj_train_data"
output_dir = r'.\CVAT_Patches'

# Get list of image and annotation paths
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
annotation_paths = [os.path.join(annotation_dir, os.path.basename(p).replace('.jpg', '.txt')) for p in image_paths]

for img_path, anno_path in zip(image_paths, annotation_paths):
    extract_patches(img_path, anno_path, output_dir)