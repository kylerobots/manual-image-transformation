import argparse
import numpy
from PIL import Image
from PIL import ImageOps
from PIL import UnidentifiedImageError
import TransformationCalculator


def loadImage(filename: str) -> numpy.ndarray:
    """!
    @brief Load an image from file to return as a numpy array.

    This uses Pillow as an intermediary to support loading a variety of image formats. After, it converts the image to
    grayscale and returns it as a numpy array.
    @param filename The file to load.
    @return numpy.ndarray A numpy array of the grayscale version of the loaded image.
    @throws FileNotFoundError Thrown if the file does not exist.
    @throws PIL.UnidentifiedImageError Thrown if the image cannot be read.
    """
    image = Image.open(filename)
    grayscale_image = ImageOps.grayscale(image)
    image_array = numpy.array(grayscale_image)
    return image_array


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
    except FileNotFoundError as ex:
        print('Unable to find {0:s}'.format(args.first_image))
        exit(-1)
    except UnidentifiedImageError:
        print('Unable to read {0:s}'.format(args.first_image))
        exit(-1)
    try:
        second_image = loadImage(args.second_image)
    except FileNotFoundError:
        print('Unable to find {0:s}'.format(args.second_image))
        exit(-1)
    except UnidentifiedImageError:
        print('Unable to read {0:s}'.format(args.second_image))
        exit(-1)
    # Pass the images in to the module for transformation prediction.
    calculator = TransformationCalculator.TransformmationCalculator()
    transform = calculator.calculateTransform(first_image, second_image)
    print('The calculated transform matrix between the two images is:')
    print(transform)
