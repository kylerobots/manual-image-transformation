import argparse
from evaluation import Evaluator
from scipy.spatial.transform import Rotation
import cv2
import features
import numpy


def loadImage(filename: str) -> numpy.ndarray:
    """!
    @brief Read an image from file and convert to grayscale.
    @param filename The image file to read.
    @return numpy.ndarray Returns the image as a numpy array.
    @raise ValueError raised if the image file cannot be found or read.
    """
    try:
        color_image = cv2.imread(filename)
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    except:
        raise ValueError('Unable to load image from {0:s}'.format(filename))
    return image


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


def loadPose(filename: str) -> numpy.ndarray:
    """!
    @brief Load the ground truth from file.

    The file should be a CSV file with a single line. All other lines will be ignored. The line should contain the
    ground truth at the time an image was captured. The pose is stored as follows:

    x,y,z,roll,pitch,yaw

    @param filename The file to read
    @return numpy.ndarray A 4x4 homogenous transform representing the pose in the file.
    @throw ValueError Thrown if the file could not be read.
    """
    pose = numpy.eye(4)
    with open(filename) as file:
        line = file.readline()
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
    return pose


if __name__ == '__main__':
    # Use command line arguments to specify which images to read.
    parser = argparse.ArgumentParser(
        description='Determine the 2D transformation that likely occurred between two images')
    parser.add_argument('first_image', metavar='first_image', type=str,
                        help='The first image to use')
    parser.add_argument('first_pose', metavar='first_pose',
                        type=str, help='The first pose file to use')
    parser.add_argument('second_image', metavar='second_image', type=str,
                        help='The second image in the sequence')
    parser.add_argument('second_pose', metavar='second_pose',
                        type=str, help='The second pose file to use')
    parser.add_argument('calibration_file', metavar='calibration_file',
                        type=str, help='The file containing camera parameters')
    args = parser.parse_args()
    # Load the data from file.
    first_image = loadImage(args.first_image)
    first_pose = loadPose(args.first_pose)
    second_image = loadImage(args.second_image)
    second_pose = loadPose(args.second_pose)
    intrinsic = loadParameters(args.calibration_file)
    # Create the evaluator with the data.
    evaluator_helper = Evaluator(first_image, first_pose,
                                 second_image, second_pose, intrinsic)
    # Create the selected detector and descriptor.
    detector = cv2.SIFT_create()
    dummy_detector = features.PointDetector(threshold=0.25)
    # Evaluate the provided detector and descriptor.
    (translation_error, rotation_error) = evaluator_helper.evaluate(detector, detector)
    print('Translational error:')
    print(translation_error)
    print('Rotational error:')
    print(rotation_error)
