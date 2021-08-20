import cv2
import numpy
from typing import List


class FAST(object):
    """!
    @brief An implementation of the FAST detection algorithm.
    """

    def __init__(self, threshold: float) -> None:
        """!
        @brief Create the detector.
        @param threshold The delta value by which the neighboring block of pixels must differ from the candidate pixel.
        """
        super().__init__()
        self._threshold = threshold

    def detect(self, image: numpy.ndarray) -> List[cv2.KeyPoint]:
        """!
        @brief Find keypoints by use of the FAST algorithm.

        This method compares a pixel to the 16 pixels that form a circle around it. The candidate pixel is considered a
        keypoint if 9 contiguous pixels on that circle are above or below the candidate pixel's intensity by an amount
        equal to the threshold. In other words, if circle_pixel >= center_pixel + threshold OR circle_pixel <=
        center_pixel - treshold.
        @param image The grayscale image to scan.
        @return List[cv2.KeyPoint] returns a list of all the keypoints that meet the above criteria.
        """
        keypoints = []
        # Iterate through each non-border pixel
        for i in range(3, image.shape[0] - 3):
            for j in range(3, image.shape[1] - 3):
                is_keypoint = self._checkPixel(image, i, j)
                if is_keypoint:
                    keypoint = cv2.KeyPoint()
                    keypoint.angle = -1
                    keypoint.octave = 0
                    keypoint.pt = [int(i), int(j)]
                    keypoint.size = 3
                    keypoints.append(keypoint)
        return keypoints

    def _checkPixel(self, image: numpy.ndarray, x: int, y: int) -> bool:
        """!
        @brief Perform the comparison of the candidate pixel and surrounding circle.

        This algorithm assumes that the entire circle of radius 3 around the target pixel exists.
        @param image The image to check
        @param x The X coordinate of the candidate pixel
        @param y The Y coordinate of the candidate pixel
        @return true if the pixel meets the criteria, false otherwise.
        @throw ValueError Thrown if the x or y values prevent evaluation of a full circle.
        """
        target_value = image[x, y]
        # Verify that the full circle can be checked.
        if x < 3 or x > image.shape[0] - 4 or y < 3 or y > image.shape[1] - 4:
            raise ValueError(
                'Target pixels must be more than 3 pixels from the image border')
        circle_values = numpy.zeros((16))
        # Per the paper, check the 4 pixels at cardinal directions
        circle_values[0] = abs(
            image[x, y - 3] - target_value) >= self._threshold
        circle_values[4] = abs(
            image[x + 3, y] - target_value) >= self._threshold
        circle_values[8] = abs(
            image[x, y + 3] - target_value) >= self._threshold
        circle_values[12] = abs(
            image[x - 3, y] - target_value) >= self._threshold
        # If three of those are not above or below the threshold, then we can stop as it isn't possible to meet the
        # keypoint criteria.
        if circle_values.sum() < 3:
            return False
        # If that criteria is met, check the rest of the points. I could recreate a circle drawing algorithm, but
        # hard coding will work okay for now.
        circle_values[1] = abs(image[x + 1, y - 3] -
                               target_value) >= self._threshold
        circle_values[2] = abs(image[x + 2, y - 2] -
                               target_value) >= self._threshold
        circle_values[3] = abs(image[x + 3, y - 1] -
                               target_value) >= self._threshold
        circle_values[5] = abs(image[x + 3, y + 1] -
                               target_value) >= self._threshold
        circle_values[6] = abs(image[x + 2, y + 2] -
                               target_value) >= self._threshold
        circle_values[7] = abs(image[x + 1, y + 3] -
                               target_value) >= self._threshold
        circle_values[9] = abs(image[x - 1, y + 3] -
                               target_value) >= self._threshold
        circle_values[10] = abs(image[x - 2, y + 2] -
                                target_value) >= self._threshold
        circle_values[11] = abs(image[x - 3, y + 1] -
                                target_value) >= self._threshold
        circle_values[13] = abs(image[x - 3, y - 1] -
                                target_value) >= self._threshold
        circle_values[14] = abs(image[x - 2, y - 2] -
                                target_value) >= self._threshold
        circle_values[15] = abs(image[x - 1, y - 3] -
                                target_value) >= self._threshold
        # Now check if there are any set of 9 contiguous values. There is probably a way to parallelize this, but this
        # will work.
        for i in range(16):
            indices = range(i, 10)
            contiguous_values = circle_values.take(indices, mode='wrap')
            if contiguous_values.sum() == 9:
                return True
        # If we reached here, then there is no contiguous region and therefore this is not a keypoint.
        return False
