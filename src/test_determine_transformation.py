import cv2
import determine_transformation
import unittest


class TestFileInput(unittest.TestCase):
    """!
    @brief Test the ability to read image files. We don't care about the resulting keypoints though.
    """

    def setUp(self) -> None:
        """!
        @brief Setup function that provides a stock detector to use when checking file handling.
        """
        self.detector = cv2.SIFT_create()
        return super().setUp()

    def test_bad_file(self):
        """!
        @test Test that the function throws correctly when unable to open a file.
        """
        # Load some other non-image file.
        self.assertRaises(ValueError, determine_transformation.processImage,
                          'data/000000.txt', self.detector, self.detector)

    def test_good_file(self):
        """!
        @test Test that the function can read in valid image files.
        """
        result = determine_transformation.processImage(
            'data/000000.png', self.detector, self.detector)
        self.assertIsInstance(result, tuple, 'Returned result is unexpected')

    def test_no_file(self):
        """!
        @test Test that the function throws correctly when a nonexistant file is specified.
        """
        self.assertRaises(ValueError, determine_transformation.processImage,
                          'data/999999.png', self.detector, self.detector)
