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
        # Only one keypoint should be found for the center pixel
        self.assertEqual(
            len(result), 1, 'PointDetector did not find only 1 keypoint.')
        self.assertTupleEqual(
            result[0].pt, (1, 1), 'PointDetector did not find the expected keypoint.')
        # Even though these won't occur in grayscale, negative values should still work.
        image = numpy.array([[0, 0, 0], [0, -255, 0], [0, 0, 0]])
        result = detector.detect(image)
        self.assertEqual(
            len(result), 1, 'PointDetector did not find only 1 keypoint.')
        self.assertTupleEqual(
            result[0].pt, (1, 1), 'PointDetector did not find the expected keypoint.')
