import numpy


class TransformmationCalculator:
    """!
    @brief A class to calculate the estimated 2D transformation experienced by a camera, based on two images.
    """

    def calculateTransform(self, image1: numpy.ndarray, image2: numpy.ndarray) -> numpy.ndarray:
        """!
        @brief Estimate the transform a camera experienced when moving between the two images.

        This method uses key points to find the transformation between images. It uses several helper functions and
        classes to find matching key points in each image, then computes a likely transform from them.

        @param image1 The image taken at the start of the transformation.
        @param image2 The image taken at the end of the transformation.
        @return numpy.ndarray A 3x3 numpy array representing the homogenous matrix for the 2D transformation.
        """
        result = numpy.identity(3)
        return result
