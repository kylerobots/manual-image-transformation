from scipy.spatial import transform
from Evaluator import Evaluator
import numpy
import unittest


class TestEvaluator(unittest.TestCase):
    """!
    @brief Test the various components of the evaluator class.
    """

    def testCalculateTranslationDiff(self):
        """!
        @test Test that translations between two transformations are calculated correctly.
        """
        evaluator = Evaluator()
        # Test Identity
        transform1 = numpy.eye(4)
        transform2 = numpy.eye(4)
        (result, _) = evaluator._calculateDifference(transform1, transform2)
        self.assertEqual(result, 0.0)
        # Test nonzero
        vector1 = numpy.array([3.0, 4.0, 5.0]).transpose()
        transform1[0:3, 3] = vector1
        vector2 = numpy.array([-10.0, -11.5, -12.75]).transpose()
        transform2[0:3, 3] = vector2
        (result, _) = evaluator._calculateDifference(transform1, transform2)
        self.assertAlmostEqual(result, 26.9130545, 6)
        # Order shouldn't matter
        (result, _) = evaluator._calculateDifference(transform2, transform1)
        self.assertAlmostEqual(result, 26.9130545, 6)

    def testCalculateRotationDiff(self):
        """!
        @test Test that rotations between two transformations are calculated correctly.
        """
        # Test identity
        evaluator = Evaluator()
        transform1 = numpy.eye(4)
        transform2 = numpy.eye(4)
        (_, result) = evaluator._calculateDifference(transform1, transform2)
        self.assertEqual(result, 0.0)
        # Test arbitrary rotation
        rot1 = numpy.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rot2 = numpy.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        transform1[0:3, 0:3] = numpy.matmul(transform1[0:3, 0:3], rot1)
        transform2[0:3, 0:3] = numpy.matmul(transform2[0:3, 0:3], rot2)
        (_, result) = evaluator._calculateDifference(transform1, transform2)
        self.assertAlmostEqual(result, 120.0 * numpy.pi / 180.0, 8)
        # Order shouldn't matter
        (_, result) = evaluator._calculateDifference(transform2, transform1)
        self.assertAlmostEqual(result, 120.0 * numpy.pi / 180.0, 8)
        # Test when the angle is pi
        transform1 = numpy.eye(4)
        transform2 = numpy.eye(4)
        transform2[0, 0] = -1.0
        transform2[1, 1] = -1.0
        (_, result) = evaluator._calculateDifference(transform1, transform2)
        # It might wrap to -pi, so check the absolute value
        self.assertAlmostEqual(abs(result), numpy.pi, 8)
