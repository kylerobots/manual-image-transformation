import cv2
import numpy
from scipy.signal import convolve2d
from typing import List


class Harris(object):
    """!
    @brief An implementation of a Harris corner detector.
    """

    def __init__(self, threshold: float) -> None:
        """!
        @brief Create the detector.
        @param threshold The value above which any corner must score to be considered a keypoint.
        """
        super().__init__()
        self._threshold = threshold
        self._k = 0.05
        # Use 3x3 Sobel filters for the gradient.
        self._x_derivative_filter = numpy.array(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self._y_derivative_filter = numpy.array(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self._sum_filter = numpy.ones((3, 3))

    def detect(self, image: numpy.ndarray) -> List[cv2.KeyPoint]:
        """!
        @brief Use the Harris corner detector algorithm to find keypoints.
        @param image The grayscale image to scan.
        @return List[cv2.KeyPoint] returns a list of all the keypoints that exceed the threshold value for the detector.
        """
        # First, compute the gradients across the entire image
        x_gradient = convolve2d(image, self._x_derivative_filter, 'same')
        y_gradient = convolve2d(image, self._y_derivative_filter, 'same')
        # Compute the M matrix, which is the structure tensor
        Mxx = convolve2d(numpy.multiply(
            x_gradient, x_gradient), self._sum_filter, 'same')
        Mxy = convolve2d(numpy.multiply(
            x_gradient, y_gradient), self._sum_filter, 'same')
        Myy = convolve2d(numpy.multiply(
            y_gradient, y_gradient), self._sum_filter, 'same')
        scores = numpy.zeros_like(image, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                M = numpy.zeros((2, 2))
                M[0, 0] = Mxx[i, j]
                M[0, 1] = Mxy[i, j]
                M[1, 0] = M[0, 1]
                M[1, 1] = Myy[i, j]
                scores[i, j] = numpy.linalg.det(M) - self._k*(M.trace()**2)
        # Now find any pixels that have local maxima scores above a certain threshold
        keypoints = []
        # Use a 3x3 window to search.
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                # There are nine total pixels in the local area. See if the center pixel has the maximum value
                local_region = scores[i-1:i+2, j-1:j+2]
                candidate_score = scores[i, j]
                max_score = numpy.max(local_region)
                if candidate_score >= self._threshold and candidate_score == max_score:
                    keypoint = cv2.KeyPoint()
                    keypoint.angle = -1
                    keypoint.octave = 0
                    keypoint.pt = [int(i), int(j)]
                    keypoint.response = candidate_score
                    keypoint.size = 3
                    keypoints.append(keypoint)
        # Convert these keypoints into the cv type
        return keypoints
