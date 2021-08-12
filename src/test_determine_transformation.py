import cv2
import numpy
import determine_transformation
import unittest


class TestFileInput(unittest.TestCase):
    """!
    @brief Test the ability to read image files. We don't care about the resulting keypoints though.
    """

    def test_bad_file(self):
        """!
        @test Test that the function throws correctly when unable to open a file.
        """
        # Load some other non-image file.
        self.assertRaises(
            ValueError, determine_transformation.loadImage, 'data/000000.txt')

    def test_good_file(self):
        """!
        @test Test that the function can read in valid image files.
        """
        result = determine_transformation.loadImage('data/000000.png')
        self.assertIsInstance(result, numpy.ndarray,
                              'Returned result is unexpected')

    def test_no_file(self):
        """!
        @test Test that the function throws correctly when a nonexistant file is specified.
        """
        self.assertRaises(
            ValueError, determine_transformation.loadImage, 'data/999999.png')
