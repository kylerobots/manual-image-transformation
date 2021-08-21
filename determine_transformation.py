import argparse
from typing import List
from evaluation import Evaluator
from scipy.spatial.transform import Rotation
import cv2
import features
import numpy


def loadImages(image_list: str) -> List[numpy.ndarray]:
    """!
    @brief Load the series of images from the provided list.

    This will go through each line in the provided text file and load the image at that file location. Paths can be
    relative or absolute. Each line should point to an image to use for data collection. The first should be the
    reference image against which all the others will compare.
    @param image_list The list of image file locations.
    @return List[numpy.ndarray] returns a list of grayscale images.
    @throw ValueError thrown if an image can't be read or converted to grayscale.
    """
    images = []
    with open(image_list) as file:
        for image_file in file:
            # Remove trailing newline characters that might screw up the import.
            image_file = image_file.strip()
            try:
                color_image = cv2.imread(image_file.strip())
                image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            except:
                raise ValueError('Unable to load the image {0:s} specified in {1:s}'.format(
                    image_file, image_list))
            images.append(image)
    return images


def loadParameters(filename: str) -> numpy.ndarray:
    """!
    @brief Read in the calibration parameters from a text file.

    Each value in the calibration matrix should be on its own line. The line numbers are as follows:

    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]

    @param filename The file to load the intrinsic matrix from.
    @return numpy.ndarray Returns a 3x3 numpy array representing the intrinsic calibration matrix.
    """
    calibration_matrix = numpy.zeros((9,))
    with open(filename) as file:
        element = 0
        for line in file:
            calibration_matrix[element] = float(line)
            element += 1
    return calibration_matrix.reshape((3, 3))


def loadPoses(pose_list: str) -> List[numpy.ndarray]:
    """!
    @brief Load the series of ground truth poses from the provided list.

    This will go through each line in the provided text file and load the poses stored in the files at that location.
    Paths can be relative or absolute. Each line should point to a text file containing pose information in the order
    specified below. The overall order of poses should be the same as provided for the images. The first pose will be
    the reference against all others will be compared.

    The individual pose files should be a CSV file with a single line. All other lines will be ignored. The line should
    contain the ground truth at the time an image was captured. The pose is stored as follows:

    x,y,z,roll,pitch,yaw

    @param pose_list The list of all pose files to read.
    @return List[numpy.ndarray] A list of 4x4 homogenous transforms representing the poses from each file.
    @throw ValueError thrown if any of the files can't be processed.
    """
    poses = []
    with open(pose_list) as file:
        for pose_file_str in file:
            # Remove trailing newline characters that might screw up the import.
            pose_file_str = pose_file_str.strip()
            pose = numpy.eye(4)
            with open(pose_file_str) as pose_file:
                line = pose_file.readline()
                values = line.split(',')
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                r = float(values[3])
                p = float(values[4])
                t = float(values[5])
                # Convert into a homogenous transform
                pose[0, 3] = x
                pose[1, 3] = y
                pose[2, 3] = z
                rotation = Rotation.from_euler(
                    'XYZ', numpy.array([r, p, t]), degrees=False)
                pose[0:3, 0:3] = rotation.as_matrix()
            poses.append(pose)
    return poses


if __name__ == '__main__':
    # Use command line arguments to specify which images to read.
    parser = argparse.ArgumentParser(
        description='Determine the 2D transformation that likely occurred between two images')
    parser.add_argument('image_list', metavar='image_list', type=str,
                        help='A file containing the list of images to use, one per line.')
    parser.add_argument('pose_list', metavar='pose_list', type=str,
                        help='A file containing the list of poses to match the images, one per line.')
    parser.add_argument('calibration_file', metavar='calibration_file',
                        type=str, help='The file containing camera parameters')
    parser.add_argument('camera_height', metavar='camera_height',
                        type=float, help='The height of the camera above the ground plane')
    args = parser.parse_args()
    # Load the data from file.
    images = loadImages(args.image_list)
    poses = loadPoses(args.pose_list)
    intrinsic = loadParameters(args.calibration_file)
    if len(images) != len(poses):
        raise ValueError('Image list and pose list are not the same size!')
    evaluator = Evaluator(images[0], poses[0], intrinsic, args.camera_height)
    # Create evaluators to test
    detectors = {}
    detectors['Point Detector'] = features.PointDetector(threshold=95)
    detectors['Harris'] = features.Harris(threshold=7.5e6)
    detectors['FAST'] = features.FAST(threshold=128)
    detectors['OpenCV SIFT'] = cv2.SIFT_create()
    descriptor = cv2.SIFT_create()
    # Iterate through each and calculate average error across all images.
    for detector_name in detectors.keys():
        detector = detectors[detector_name]
        total_translation_error = 0.0
        total_rotation_error = 0.0
        good = True
        for i in range(1, len(images)):
            evaluator.setComparisionImage(images[i], poses[i])
            try:
                (translation_error, rotation_error) = evaluator.evaluate(
                    detector, descriptor)
            except Exception as e:
                good = False
                break
            total_translation_error += abs(translation_error)
            total_rotation_error += abs(rotation_error)
        total_translation_error /= (len(images) - 1.0)
        total_rotation_error /= (len(images) - 1.0)
        if good:
            print('{0:s}: {1:0.3f}  --  {2:0.3f}'.format(detector_name,
                  total_translation_error, total_rotation_error))
        else:
            print('{0:s}: N/A'.format(detector_name))
