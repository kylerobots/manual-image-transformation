from Harris import Harris
import numpy
import unittest


class TestHarris(unittest.TestCase):
    """!
    @brief Test the Harris detector class.
    """

    def test_detect(self):
        """!
        @test Test that the detect function works as expected.
        """
        # Create a simple array to check.
        image = numpy.zeros((10, 10))
        image[5:10, 5:10] = 255
        detector = Harris(threshold=10000)
        results = detector.detect(image)
        self.assertGreaterEqual(len(results), 1)
