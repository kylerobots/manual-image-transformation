# Comparison of Feature Detection Methods for Ground Texture Localization #
Manual implementation of feature detectors and an evaluation pipeline for comparison in ground texture localization for
CPE 645.

## Dependencies ##
There is a Pipfile that describes the required dependencies. To create a virtual environment with everything you need,
just run to following from the project root directory.
```python
pip install --upgrade pipenv
pipenv install
```
If you would like to install manually, you need:
1. NumPy
2. SciPy
3. OpenCV

## Data Format ##
The localization tests compare the different detectors across a large number of images and associated ground truth poses
for the camera when the images were captured. The data is referenced by two summary text files, one for images and one
for poses. These files list the file paths and names of each image and pose to be used by the system. The files are
listed one per line and can be relative or absolute paths from wherever you choose to run the script.

The images referenced in the image summary file are simple image files in any format readable by OpenCV. The pose files
associated with the images should be a single line text file with the 3D pose of the camera listed in a comma-delimited
format like so:

```
x,y,z,roll,pitch,yaw
```

When creating new data, the system compares all images and poses to the first in the series. So there should be at least
a little overlap between each image and the first.

Lastly, there should also be a file containing the intrinsic parameters for the camera. The 3x3 matrix of parameters is
stored with one element per line in row major order. An example of all these files is included with the source code.
The images, poses, and camera parameters are located in the data subfolder, while the two summary files are at the root
of the project called *images.txt* and *poses.txt*.

## Running the Code ##
To run the evaluation, run the following from the root of the package:
```python
python determine_transformation.py image_list pose_list calibration_file camera_height
```
Each required argument is described below. There is also a help flag, ```-h``` or ```--help``` that prints a reminder.

| Argument Name | Data Type | Description |
| --- | --- | --- |
| image_list | String | A file containing the list of images to use, one per line |
| pose_list | String | A file containing the list of poses to match the images, one per line |
| calibration_file | String | The file containing camera parameters |
| camera_height | String | The height of the camera above the ground plane |

It will read in the data, run the different detectors on the images, and calculate an average translation and rotation
error of the estimates produced using each detector. The translation errors are simply the average Euclidean distance
between the translation estimated by this framework and the ground truth translation reported in the data files. The
rotation error is the angle component of the axis-angle representation of the rotation between the estimated orientation
and ground truth orientation. Note that in a purely 2D rotation, this evaluates to the difference in angles. After
calculating these two errors, the results are printed to the console like so:
```
Detector Name 1: Translation Accuracy  --  Rotation Accuracy
Detector Name 2: Translation Accuracy  --  Rotation Accuracy
...
```