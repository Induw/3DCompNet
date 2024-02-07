import glob
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Load file paths
print("...getting file paths using glob...")
file_paths_raw = glob.glob("/Users/ravinduhettiarachchi/Documents/FYP/DataSet/**/**/PROCESSED/MPRAGE/T88_111/*t88_gfc.img")
file_paths_segmented = glob.glob("/Users/ravinduhettiarachchi/Documents/FYP/DataSet/**/**/FSL_SEG/*.img")
#print(file_paths_raw)
print(file_paths_segmented)
print(len(file_paths_raw))
print(len(file_paths_segmented))
print("file paths..loaded..")
# Step 2: Load images using nibabel
raw_images = [nib.load(file).get_fdata() for file in file_paths_raw]
segmented_images = [nib.load(file).get_fdata() for file in file_paths_segmented]
print("printing lengths..")
print(len(raw_images))
print(len(segmented_images))


# Step 3: Arrays for images
# Assuming raw_images and segmented_images are your required arrays
print("splitting the data")
# Step 4: Split the data
x_train, x_test, y_train, y_test = train_test_split(raw_images, segmented_images, test_size=0.2)
print("saving..")
# Step 5: Save the split data using numpy
np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
