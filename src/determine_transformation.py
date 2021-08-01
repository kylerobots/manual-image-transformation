import argparse
import numpy
from PIL import Image
from PIL import UnidentifiedImageError


def loadImage(filename: str) -> numpy.ndarray:
    """!
    @brief Load an image from file to return as a numpy array.

    This uses Pillow as an intermediary to support loading a variety of image formats. After, it simply converts to a
    numpy array to use by the user.
    @param filename The file to load.
    @return numpy.ndarray A numpy array of the loaded image.
    @throws FileNotFoundError Thrown if the file does not exist.
    @throws PIL.UnidentifiedImageError Thrown if the image cannot be read.
    """
    image = Image.open(filename)
    image_array = numpy.array(image)
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
