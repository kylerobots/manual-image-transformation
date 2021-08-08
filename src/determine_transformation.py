import argparse
import cv2
import numpy
import TransformationCalculator


def loadImage(filename: str) -> numpy.ndarray:
    """!
    @brief Load an image from file to return as a numpy array.

    This uses OpenCV as an intermediary to support loading a variety of image formats. After, it converts the image to
    grayscale and returns it as a numpy array.
    @param filename The file to load.
    @return numpy.ndarray A numpy array of the grayscale version of the loaded image.
    @throws ValueError Thrown if the file can't be read or the image can't be converted to grayscale.
    """
    image = cv2.imread(filename)
    if image is None:
        raise ValueError('Unable to read image from {0:s}'.format(filename))
    try:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        raise ValueError(
            'Unable to convert image to grayscale: {0:s}'.format(filename))
    return grayscale_image


if __name__ == '__main__':
    # Use command line arguments to specify which images to read.
    parser = argparse.ArgumentParser(
        description='Determine the 2D transformation that likely occurred between two images')
    parser.add_argument('first_image', metavar='1', type=str,
                        help='The first image in the sequence')
    parser.add_argument('second_image', metavar='2', type=str,
                        help='The second image in the sequence')
    args = parser.parse_args()
    # Try to load each image. If there is a problem with any of them, warn the user and exit.
    try:
        first_image = loadImage(args.first_image)
    except ValueError as ex:
        print('Unable to load {0:s}'.format(args.first_image))
        exit(-1)
    try:
        second_image = loadImage(args.second_image)
    except ValueError:
        print('Unable to load {0:s}'.format(args.second_image))
        exit(-1)
    # Pass the images in to the module for transformation prediction.
    calculator = TransformationCalculator.TransformmationCalculator()
    transform = calculator.calculateTransform(first_image, second_image)
    print('The calculated transform matrix between the two images is:')
    print(transform)
