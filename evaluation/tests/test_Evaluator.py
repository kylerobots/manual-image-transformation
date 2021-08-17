import cv2
from evaluation import Evaluator
import numpy
import unittest


class TestEvaluator(unittest.TestCase):
    """!
    @brief Test the various components of the evaluator class.
    """

    def setUp(self) -> None:
        """!
        @brief Set up the Evaluator with some fake data.
        """
        fake_image = numpy.zeros(100)
        fake_pose = numpy.eye(4)
        fake_intrinsic = numpy.eye(3)
        self.evaluator = Evaluator(
            fake_image, fake_pose, fake_image, fake_pose, fake_intrinsic)
        return super().setUp()

    def testCalculateRotationDiff(self):
        """!
        @test Test that rotations between two transformations are calculated correctly.
        """
        # Test identity
        transform1 = numpy.eye(4)
        transform2 = numpy.eye(4)
        (_, result) = self.evaluator._calculateDifference(transform1, transform2)
        self.assertEqual(result, 0.0)
        # Test arbitrary rotation
        rot1 = numpy.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rot2 = numpy.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        transform1[0:3, 0:3] = numpy.matmul(transform1[0:3, 0:3], rot1)
        transform2[0:3, 0:3] = numpy.matmul(transform2[0:3, 0:3], rot2)
        (_, result) = self.evaluator._calculateDifference(transform1, transform2)
        self.assertAlmostEqual(result, 120.0 * numpy.pi / 180.0, 8)
        # Order shouldn't matter
        (_, result) = self.evaluator._calculateDifference(transform2, transform1)
        self.assertAlmostEqual(result, 120.0 * numpy.pi / 180.0, 8)
        # Test when the angle is pi
        transform1 = numpy.eye(4)
        transform2 = numpy.eye(4)
        transform2[0, 0] = -1.0
        transform2[1, 1] = -1.0
        (_, result) = self.evaluator._calculateDifference(transform1, transform2)
        # It might wrap to -pi, so check the absolute value
        self.assertAlmostEqual(abs(result), numpy.pi, 8)
        # Test an extreme value
        transform2 = -1.0 * numpy.eye(4)
        (_, result) = self.evaluator._calculateDifference(transform2, transform1)
        self.assertAlmostEqual(abs(result), numpy.pi)

    def testCalculateTransform(self):
        """!
        @test Test that the class extracts the right transform from two adjacent sets of points.
        """
        # Create some points in the first frame.
        z = 1.0
        first_points = numpy.array(
            [[0, 0, z], [2, 0, z], [2, 5, z], [0, 5, z]], dtype=numpy.float32)
        # Create a transformation that will move the camera
        R = numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = numpy.array([[3.0], [-5.0], [0.0]])
        expected_result = numpy.eye(4)
        expected_result[0:3, 0:3] = R
        expected_result[0:3, 3:] = t
        # Determine where the second points would be given that.
        second_points = (numpy.matmul(
            R, first_points.transpose()) + t).transpose()
        # Create a simple intrinsic matrix to project onto a fictional camera
        intrinsic = numpy.array(
            [[1.0, 0.0, 20.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
        # Use no distortion or transformations
        rvec = numpy.zeros((3, 1))
        tvec = rvec
        distortion = numpy.zeros((5, 1))
        # Project the points into the camera
        (camera_first_points, _) = cv2.projectPoints(
            first_points, rvec, tvec, intrinsic, distortion)
        camera_first_points = camera_first_points.squeeze()
        (camera_second_points, _) = cv2.projectPoints(
            second_points, rvec, tvec, intrinsic, distortion)
        camera_second_points = camera_second_points.squeeze()
        # Using these projected points, can the object recover the correct initial transform
        result = self.evaluator._calculateTransform(
            camera_first_points, camera_second_points, intrinsic)
        # The matrix comparisions aren't reliable near zero, so check elements manually.
        for i in range(expected_result.shape[0]):
            for j in range(expected_result.shape[1]):
                result_element = result[i, j]
                expected_element = expected_result[i, j]
                self.assertAlmostEqual(result_element, expected_element, 6,
                                       'Matrix element ({0:d}, {1:d}) is incorrect.'.format(i, j))

    def testCalculateTranslationDiff(self):
        """!
        @test Test that translations between two transformations are calculated correctly.
        """
        # Test Identity
        transform1 = numpy.eye(4)
        transform2 = numpy.eye(4)
        (result, _) = self.evaluator._calculateDifference(transform1, transform2)
        self.assertEqual(result, 0.0)
        # Test nonzero
        vector1 = numpy.array([3.0, 4.0, 5.0]).transpose()
        transform1[0:3, 3] = vector1
        vector2 = numpy.array([-10.0, -11.5, -12.75]).transpose()
        transform2[0:3, 3] = vector2
        (result, _) = self.evaluator._calculateDifference(transform1, transform2)
        self.assertAlmostEqual(result, 26.9130545, 6)
        # Order shouldn't matter
        (result, _) = self.evaluator._calculateDifference(transform2, transform1)
        self.assertAlmostEqual(result, 26.9130545, 6)

    @unittest.skip('Skip until more intense debugging can occur in an appropriate branch.')
    def testFindCorrespondence(self):
        """!
        @test Test that the matching between keypoints is correct.
        """
        # Create some dummy keypoints and descriptors to match. Make the descriptors really far apart to be sure.
        keypoints1 = []
        descriptors1 = numpy.zeros(shape=(3, 1))
        keypoint = cv2.KeyPoint()
        for i in range(3):
            keypoint.pt = (float(i), 0.0)
            keypoints1.append(keypoint)
            descriptors1[i] = i*100.0
        keypoints2 = []
        descriptors2 = numpy.zeros(shape=(5, 1))
        for i in range(5):
            keypoint.pt = (0.0, float(i))
            keypoints2.append(keypoint)
            descriptors2[i] = i*105.0
        (first_points, second_points) = self.evaluator._findCorrespondence(
            keypoints1, descriptors1, keypoints2, descriptors2)
        expected_first = numpy.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        expected_second = numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        self.assertTrue(numpy.array_equal(first_points, expected_first))
        self.assertTrue(numpy.array_equal(second_points, expected_second))
