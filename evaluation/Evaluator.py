import cv2
import numpy
from typing import Any
from typing import List
from typing import Tuple
from scipy.spatial.transform import Rotation


class Evaluator:
    """!
    @brief A class to help evaluate different feature detectors and descriptors.

    This class uses the provided images, detector, and descriptor to determine an estimated transformation between two
    images. It then calculates an error based on provided ground truth information. By doing this, it serves to allow
    the testing of a variety of different detectors and descriptors to see works the best.
    """

    def __init__(self, reference_image: numpy.ndarray, reference_pose: numpy.ndarray, intrinsic: numpy.ndarray, camera_height: float) -> None:
        """!
        @brief Construct the object with the given settings.

        @param reference_image The first image to use in comparision.
        @param reference_pose A 4x4 homogenous transform representing the ground truth of the camera when the first image
        is captured.
        @param intrinsic A 3x3 matrix representing the camera's intrinsic parameters.
        @param camera_height The height of the camera above the ground plane.
        """
        # Store all the values.
        self._first_image = reference_image
        self._first_pose = reference_pose
        self._intrinsic = intrinsic
        self.camera_height = camera_height
        # Used to find correspondences between two sets of keypoints by checking if one candidate is lower than this
        # percentage of the next closest candadate. i.e. match if distance between A and X is less than
        # descriptor_match_threshold * distance between A and Y.
        self._descriptor_match_threshold = 0.7
        pass

    def evaluate(self, detector: Any, descriptor: Any) -> Tuple[float, float]:
        """!
        @brief Use the provided detector and descriptor to estimate the transformation and compute the error.

        This method will use the objects to first detect keypoints in both target images, then describe those keypoints.
        Helper functions will then estimated the transformation likely to occur between the two images. This is compared
        to the provided ground truth to determine the error values.

        @param detector A detector to use evaluate. It should provide a method called 'detect' that receives an image,
        stored as a numpy.ndarray, and should return a list of cv2.Keypoints.
        @param descriptor A descriptor to evaluate. It should provide a method called 'compute' that has arguments of:
        1) A numpy.ndarray representing an image, and 2) a list of cv2.Keypoints for that image. It should return a
        tuple consisting of the keypoints provided to it and an NxM numpy array, where N is the number of keypoints and
        M is the size of the descriptor. This is to maintain compatability with OpenCV's implementation.
        @return Tuple[float, float] returns the translational and rotational errors
        """
        # First, generate keypoints and descriptors for each image
        first_keypoints = detector.detect(self._first_image)
        (_, first_descriptors) = descriptor.compute(
            self._first_image, first_keypoints)
        second_keypoints = detector.detect(self._second_image)
        (_, second_descriptors) = descriptor.compute(
            self._second_image, second_keypoints)
        # Then, match the best fitting keypoints in each image
        (first_points, second_points) = self._findCorrespondence(
            first_keypoints, first_descriptors, second_keypoints, second_descriptors)
        # Use these to compute the transformation
        transform = self._calculateTransform(
            first_points, second_points, self._intrinsic)
        # Compare against the ground truth.
        results = self._calculateDifference(transform, self._expectedTransform)
        return results

    def setComparisionImage(self, image: numpy.ndarray, pose: numpy.ndarray) -> None:
        """!
        @brief Set the second image and associated pose for error calculation.

        This also determines the expected transformation that @ref evaluate should determine in an ideal case.
        @param image The image to compare to the one set at construction.
        @param pose The 4x4 homogenous transform representing the pose associated with the image.
        """
        self._second_image = image
        self._second_pose = pose
        # The two poses can be used to calculate what the final transformation between the images should be.
        self._expectedTransform = numpy.matmul(
            self._first_pose.transpose(), pose)

    def _calculateDifference(self, first: numpy.ndarray, second: numpy.ndarray) -> Tuple[float, float]:
        """!
        @brief Calculate the translational and rotational difference between two transformations.

        To calculate, the translational difference is just the Euclidean distance between the two translation vectors of
        the transformation. The rotational error is calculated by finding the rotation between the two transformations,
        since they also represent frames of reference. This rotation is then converted to axis angle format. Since that
        format involves a unit axis and a single rotation about that axis, the angle serves as a measure of how far
        apart the two transformations are.

        @param first The first transformation to use. It should be a 4x4 array.
        @param second The second transformation to use. It should be a 4x4 array.
        @return tuple[float, float] Returns the translational and rotational error between the two transforms.
        """
        # Extract the translations and rotations from each homogenous matrix.
        translation1 = first[0:3, 3]
        translation2 = second[0:3, 3]
        rotation1 = first[0:3, 0:3]
        rotation2 = second[0:3, 0:3]
        # Calculate the norm for the translation difference
        translation_difference = numpy.linalg.norm(translation1 - translation2)
        # Determine the axis-angle representation of the rotations
        rotation_matrix = numpy.matmul(rotation1, rotation2.transpose())
        rotation_matrix_scipy = Rotation.from_matrix(rotation_matrix)
        rotation_vector = rotation_matrix_scipy.as_rotvec()
        rotation_difference = numpy.linalg.norm(rotation_vector)
        return (translation_difference, rotation_difference)

    def _calculateTransform(self, first: numpy.ndarray, second: numpy.ndarray, intrinsic: numpy.ndarray) -> numpy.ndarray:
        """!
        @brief Given two sets of matching points seen by a camera, estimate the transformation the camera underwent
        between the two sets.

        This uses two OpenCV methods to calculate the result. First is a RANSAC based method to find the homography
        between the two sets of points. Then, it uses the camera parameters to decompose the homography into its
        components, including the rotation and translation vectors.

        This produces 4 possible results. Using knowledge of the world (points in front of the camera, camera above
        the ground plane, etc.), only one result is selected. This is what is returned.

        @param first Nx2 matrix representing the locations of the matched keypoints in the first image.
        @param second Nx2 matrix representing the locations of the matched keypoints in the second image.
        @param intrinsic The 3x3 intrinsic matrix of the camera.
        @return numpy.ndarray Returns a 4x4 matrix of the homogenous transformation the camera underwent between the two
        views that produced the given points.
        """
        # Estimate the homography
        H, _ = cv2.findHomography(first, second, cv2.RANSAC)
        # Determine candidate transformations
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(
            H, intrinsic)
        # Find the correct one by comparision of the normals to identity
        normal_cos_angles = numpy.zeros((num_solutions, 1))
        unit_vector = numpy.array([0.0, 0.0, 1.0])
        for i in range(num_solutions):
            # The normal vector should already be normalized, so no need to normalize.
            normal_cos_angles[i] = numpy.dot(normals[i].squeeze(), unit_vector)
        # The right one to pick is whichever is the cos(angle) closest to 1 AKA most aligned.
        selection = numpy.argmax(normal_cos_angles)
        transformation = numpy.eye(4)
        transformation[0:3, 0:3] = rotations[selection]
        # The homography decomposes to a scale factor based on distance from the camera. But since we know the camera
        # height, this can be corrected for.
        transformation[0:3, 3:] = translations[selection] * self.camera_height
        return transformation

    def _findCorrespondence(self, first_keypoints: List[cv2.KeyPoint], first_descriptors: numpy.ndarray, second_keypoints: List[cv2.KeyPoint], second_descriptors: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """!
        @brief Use descriptors to find matching keypoints in two sets.

        This matches keypoints between two sets by comparing their associated descriptors. It uses a K-Nearest Neighbor
        object to identify 2 possible matches between a given keypoint from the first set and all keypoints in the
        second set. With each candidate match, it also provides a distance measure used to quantify how close of a match
        the two are. A match is considered correct if its distance measure is less than a certain fraction of the next
        closest match for that keypoint.

        In other words, say there is a keypoint in the first group, A. The KNN object returns candidate matches to
        the two most similar keypoints in the second group, X and Y. A and X are considered a match if the distance
        between X and A is less than 70% of the distance between A and Y.

        @param first_keypoints A list of keypoints from the first image.
        @param first_descriptors The array containing the descriptors for these keypoints.
        @param second_keypoints A list of keypoints from the second image.
        @param second_descriptors The array containing the descriptors for these keypoints.
        @return tuple[numpy.ndarray, numpy.ndarray] Returns two matching arrays, each Nx2 where N are the number of
        matched keypoints. Each element[i, :] measures the pixel value of the keypoint in one or the other array.
        """
        # First, create a KNN matcher to find neighbors
        matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_FLANNBASED)
        # Find the candidate matches
        knn_matches = matcher.knnMatch(
            first_descriptors, second_descriptors, 2)
        # Find which are confirmed matches
        good_matches = []
        for match_set in knn_matches:
            # Compare the first match to the second match to see if it meets the criteria.
            if match_set[0].distance < self._descriptor_match_threshold * match_set[1].distance:
                # Mark this off as a correct match if so
                good_matches.append(match_set[0])
        # Now that all matches are found, create the arrays of keypoint locations
        first_points = numpy.zeros(shape=(len(good_matches), 2))
        second_points = numpy.zeros_like(first_points)
        for i, match in enumerate(good_matches):
            # The match objects only provide which index in each keypoint list is a match. Use these to look up the
            # position information for the given keypoint.
            matched_keypoint1 = first_keypoints[match.queryIdx]
            matched_keypoint2 = second_keypoints[match.trainIdx]
            first_points[i][0] = matched_keypoint1.pt[0]
            first_points[i][1] = matched_keypoint1.pt[1]
            second_points[i][0] = matched_keypoint2.pt[0]
            second_points[i][1] = matched_keypoint2.pt[1]
        return (first_points, second_points)
