from numpy import ndarray
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
