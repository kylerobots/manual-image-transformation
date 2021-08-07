import numpy


class PointDetector(object):
    """!
    @brief A class to perform 2nd derivative point detection, as described in the lecture notes.
    """

    def __init__(self, threshold: float) -> None:
        """!
        @brief Construct a PointDetector object.

        @param threshold The threshold value to use to find points. All pixels greater than this will be labeled. This
        value should be a positive number.
        """
        super().__init__()
        if threshold <= 0.0:
            raise ValueError('Threshold must be a number greater than zero.')
        self.threshold = threshold

    def detect(self, image: numpy.ndarray) -> numpy.ndarray:
        """!
        @brief Perform point detection using a 2nd order derivative filter.

        This uses the filter defined in class:

        [-1 -1 -1; -1 8 -1; -1 -1 -1]

        to identify keypoints. The result is then compared against the value provided in threshold to see which points
        qualify.

        @return numpy.ndarray Returns an array holding the keypoints. The array is Nx2 where N is the number of
        detected keypoints. The values correspond to the x and y pixel locations of these keypoints.
        """
        return numpy.zeros((1, 2))
