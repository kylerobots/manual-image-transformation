from features import FAST
import numpy
import unittest


class TestFAST(unittest.TestCase):
    """!
    @brief Test the Harris detector class.
    """

    def test_detect(self):
        """!
        @test Test that the detect function works as expected.
        """
        # Create a simple array to check.
        image = numpy.zeros((10, 10))
        # This point should be detected
        image[5, 5] = 255
        # This should not because it is too close to the border
        image[0, 0] = 255
        detector = FAST(threshold=1)
        results = detector.detect(image)
        self.assertEqual(len(results), 1, 'Incorrect number of keypoints.')
        self.assertEqual(results[0].pt[0], 5,
                         'Incorrect X coordinate of keypoint.')
        self.assertEqual(results[0].pt[1], 5,
                         'Incorrect Y coordinate of keypoint.')
        # It works the same if the intensities are reversed.
        image = numpy.ones_like(image) * 255
        image[5, 5] = 0
        results = detector.detect(image)
        self.assertEqual(len(results), 1, 'Incorrect number of keypoints.')
        self.assertEqual(results[0].pt[0], 5,
                         'Incorrect X coordinate of keypoint.')
        self.assertEqual(results[0].pt[1], 5,
                         'Incorrect Y coordinate of keypoint.')
