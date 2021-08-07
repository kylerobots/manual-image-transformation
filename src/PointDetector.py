import cv2
import numpy
from scipy import signal


class PointDetector(object):
    """!
    @brief A class to perform 2nd derivative point detection, as described in the lecture notes.
    """

    def __init__(self, threshold: float) -> None:
        """!
        @brief Construct a PointDetector object.

        @param threshold The threshold value to use to find points. All pixels greater than or equal to this will be
        labeled. This value should be a positive number.
        """
        super().__init__()
        if threshold <= 0.0:
            raise ValueError('Threshold must be a number greater than zero.')
        self.threshold = threshold

    def detect(self, image: numpy.ndarray) -> list:
        """!
        @brief Perform point detection using a 2nd order derivative filter.

        This uses the filter defined in class:

        [-1 -1 -1; -1 8 -1; -1 -1 -1]

        to identify keypoints. The result is then compared against the value provided in threshold to see which points
        qualify.

        @return numpy.ndarray Returns an array holding the keypoints. The array is Nx2 where N is the number of
        detected keypoints. The values correspond to the x and y pixel locations of these keypoints.
        """
        filter = numpy.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # Run the convolution to find the 2nd derivative
        result = signal.convolve2d(in1=image, in2=filter, mode='same')
        threshold_result = numpy.absolute(result) >= self.threshold
        keypoints = numpy.argwhere(threshold_result)
        return self._createCVKeypoints(keypoints)

    def _createCVKeypoints(self, keypoint_array: numpy.ndarray) -> list:
        """!
        @brief Convert the numpy arrays of keypoint locations into a list of OpenCV keypoints.

        This is for convenience when performing the transformation calculation.

        @param keypoint_array A numpy array of keypoints, Nx2 in size, where the elements are the pixel coordinates of the
        keypoints.
        @return list Returns a list of cv2.Keypoints.
        """
        keypoints = []
        for i in range(keypoint_array.shape[0]):
            x = keypoint_array[i, 0]
            y = keypoint_array[i, 1]
            keypoint = cv2.KeyPoint(float(x), float(y), size=1.0)
            keypoints.append(keypoint)
        return keypoints
