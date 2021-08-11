import argparse
import cv2
import numpy
from scipy.spatial.transform.rotation import Rotation

from Evaluator import Evaluator


def loadExpectedResult(filename: str) -> numpy.ndarray:
    expected_transformation = numpy.eye(4)
    with open(filename) as file:
        # Take only the first three lines as the x, y, and theta coordinates
        expected_transformation[0, 3] = float(file.readline())
        expected_transformation[1, 3] = float(file.readline())
        theta = float(file.readline())
        scipy_rotation = Rotation.from_euler('z', theta, degrees=False)
        expected_transformation[0:3, 0:3] = scipy_rotation.as_matrix()
    return expected_transformation


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


def processImage(filename: str, detector: cv2.Feature2D, descriptor: cv2.Feature2D) -> tuple[list[cv2.KeyPoint], numpy.ndarray]:
    """!
    @brief Load an image, detect keypoints within it, and describe those keypoints.

    @param filename The file to load the image from.
    @param detector The keypoint detector to use.
    @param descriptor The keypoint describer to use.
    @return tuple A tuple containing the list of keypoints and list of descriptions.
    """
    try:
        # Load the image first
        color_image = cv2.imread(filename)
        # Convert to grayscale
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # Detect keypoints
        keypoints = detector.detect(image)
        # Describe the keypoints
        (_, descriptions) = descriptor.compute(image, keypoints)
    except:
        # I don't really care about particular errors, just if the image can go through the whole process or not.
        raise ValueError(
            'Unable to process image for file {0:s}'.format(filename))
    return (keypoints, descriptions)


if __name__ == '__main__':
    # Use command line arguments to specify which images to read.
    parser = argparse.ArgumentParser(
        description='Determine the 2D transformation that likely occurred between two images')
    parser.add_argument('first_image', metavar='1', type=str,
                        help='The first image in the sequence')
    parser.add_argument('second_image', metavar='2', type=str,
                        help='The second image in the sequence')
    parser.add_argument('calibration_file', metavar='calibration_file',
                        type=str, help='The file containing camera parameters')
    parser.add_argument('expected_result_file', metavar='expected_result_file', type=str,
                        help='The file containing the ground truth 2D transformation between images.')
    args = parser.parse_args()

    evaluator = Evaluator()

    # Create the selected detector and describer
    detector = cv2.SIFT_create()
    # Get the keypoints and descriptions from each specified image file.
    (first_keypoints, first_descriptions) = processImage(
        args.first_image, detector, detector)
    (second_keypoints, second_descriptions) = processImage(
        args.second_image, detector, detector)

    (first, second) = evaluator._findCorrespondence(first_keypoints,
                                                    first_descriptions, second_keypoints, second_descriptions)
    # (first_points, second_points) = findCorrespondence(
    #     first_keypoints, first_descriptions, second_keypoints, second_descriptions)
    # camera_parameters = loadParameters(args.calibration_file)
    # transformation = computeTransform(
    #     first_points, second_points, camera_parameters)
    # expected_transformation = loadExpectedResult(args.expected_result_file)
    # (translation_error, rotation_error) = calculateError(
    #     transformation, expected_transformation)
    # print('Translational error:')
    # print(translation_error)
    # print('Rotational error:')
    # print(rotation_error)
