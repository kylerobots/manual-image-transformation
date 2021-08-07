from numpy import ndarray
import numpy
from PointDetector import PointDetector
import unittest


class TestPointDetector(unittest.TestCase):
    """!
    @brief Test the PointDetector class.
    """

    def test_initialize(self):
        """!
        @test Test that the class only accepts positive thresholds.
        """
        # This should be the only good one.
        detector = PointDetector(1)
        # These should all raise exceptions
        self.assertRaises(ValueError, PointDetector, 0.0)
        self.assertRaises(ValueError, PointDetector, -1.34)

    def test_detect(self):
        """!
        @test Test that the detect function works as expected.
        """
        # Create a simple array to check.
        image = numpy.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]])
        detector = PointDetector(500)
        result = detector.detect(image)
        expected_result = numpy.array([[1, 1]])
        self.assertTrue(numpy.array_equal(result, expected_result),
                        'PointDetector does not find keypoint correctly.')
        # Even though these won't occur in grayscale, negative values should still work.
        image = numpy.array([[0, 0, 0], [0, -255, 0], [0, 0, 0]])
        result = detector.detect(image)
        self.assertTrue(numpy.array_equal(result, expected_result),
                        'PointDetector does not find keypoints of negative values correctly.')
