from numpy import ndarray
import determine_transformation
import unittest


class TestFileInput(unittest.TestCase):
    """!
    @brief Test the ability to read image files.
    """

    def test_bad_file(self):
        """!
        @test Test that the function throws correctly when unable to open a file.
        """
        # Load some other non-image file.
        self.assertRaises(ValueError,
                          determine_transformation.loadImage, 'data/000000.txt')

    def test_good_file(self):
        """!
        @test Test that the function can read in valid image files.
        """
        result = determine_transformation.loadImage('data/000000.png')
        self.assertIsInstance(
            result, ndarray, 'Returned image is not a numpy array.')

    def test_grayscale(self):
        """!
        @test Test that returned images only have one channel.
        """
        result = determine_transformation.loadImage('data/000000.png')
        self.assertEqual(len(result.shape), 2,
                         'Returned image has multiple channels.')

    def test_no_file(self):
        """!
        @test Test that the function throws correctly when a nonexistant file is specified.
        """
        self.assertRaises(
            ValueError, determine_transformation.loadImage, 'data/999999.png')
