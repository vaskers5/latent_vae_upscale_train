import os
import pandas as pd

# Create the clear_df_normal_images DataFrame
df = pd.read_csv('result_cleared_df.csv')
validate_df = df[df['stage'] == 'validate']
clear_df = validate_df.dropna(subset=['width', 'size_bytes', 'height'])
clear_df_normal_images = clear_df[clear_df['size_bytes'] < 10 * 1024 * 1024]
clear_df_normal_images = clear_df.loc[~((clear_df.height +clear_df.width) > 10000)]
clear_df_normal_images.to_csv('clear_images.csv', index=False)

# Load the CSV with paths to keep
clear_df = pd.read_csv('clear_images.csv')
keep_paths = set(clear_df['path'])

# Define image extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Dataset folder
dataset_folder = 'unpacked_original_ds'

# Collect all image paths in the dataset folder
all_images = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            full_path = os.path.abspath(os.path.join(root, file))
            all_images.append(full_path)

# Find images to delete
to_delete = [img for img in all_images if img not in keep_paths]

print(f"Found {len(all_images)} images in dataset. Keeping {len(keep_paths)} images. Deleting {len(to_delete)} images.")

# Confirm before deleting (optional, but safe)
if to_delete:
    confirm = input("Proceed with deletion? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Deletion cancelled.")
        exit()

    for img in to_delete:
        try:
            os.remove(img)
            print(f"Deleted: {img}")
        except OSError as e:
            print(f"Error deleting {img}: {e}")
else:
    print("No images to delete.")
