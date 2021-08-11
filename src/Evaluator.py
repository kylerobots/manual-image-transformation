import cv2
import numpy
from typing import Any
from scipy.spatial.transform import Rotation


class Evaluator:
    """!
    @brief A class to help evaluate different feature detectors and descriptors.

    This class uses the provided images, detector, and descriptor to determine an estimated transformation between two
    images. It then calculates an error based on provided ground truth information. By doing this, it serves to allow
    the testing of a variety of different detectors and descriptors to see works the best.
    """

    def __init__(self) -> None:
        # Used to find correspondences between two sets of keypoints by checking if one candidate is lower than this
        # percentage of the next closest candadate. i.e. match if distance between A and X is less than
        # descriptor_match_threshold * distance between A and Y.
        self.descriptor_match_threshold = 0.7
        pass

    def evaluate(self, detector: Any, descriptor: Any) -> tuple[float, float]:
        """!
        @brief Use the provided detector and descriptor to estimate the transformation and compute the error.

        This method will use the objects to first detect keypoints in both target images, then describe those keypoints.
        Helper functions will then estimated the transformation likely to occur between the two images. This is compared
        to the provided ground truth to determine the error values.

        @param detector A detector to use evaluate. It should provide a method called 'detect' that will return a list
        of cv2.Keypoints from an image.
        @param descriptor A descriptor to evaluate. It should provide a method called 'compute' that will return a list
        of numpy.ndarray for each keypoint.
        """
        return (0.0, 0.0)

    def _calculateDifference(self, first: numpy.ndarray, second: numpy.ndarray) -> tuple[float, float]:
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
        # Find the correct one by knowledge of the data collection
        transformation = numpy.eye(4)
        transformation[0:3, 0:3] = rotations[0]
        transformation[0:3, 3:] = translations[0]
        return transformation

    def _findCorrespondence(self, keypoints1: list[cv2.KeyPoint], descriptors1: numpy.ndarray, keypoints2: list[cv2.KeyPoint], descriptors2: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
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

        @param keypoints1 A list of keypoints from the first image.
        @param descriptors1 The array containing the descriptors for these keypoints.
        @param keypoints2 A list of keypoints from the second image.
        @param descriptors2 The array containing the descriptors for these keypoints.
        @return tuple[numpy.ndarray, numpy.ndarray] Returns two matching arrays, each Nx2 where N are the number of
        matched keypoints. Each element[i, :] measures the pixel value of the keypoint in one or the other array.
        """
        # First, create a KNN matcher to find neighbors
        matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_FLANNBASED)
        # Find the candidate matches
        knn_matches = matcher.knnMatch(
            queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
        # Find which are confirmed matches
        good_matches = []
        for match_set in knn_matches:
            # Compare the first match to the second match to see if it meets the criteria.
            if match_set[0].distance < self.descriptor_match_threshold * match_set[1].distance:
                # Mark this off as a correct match if so
                good_matches.append(match_set[0])
        # Now that all matches are found, create the arrays of keypoint locations
        first_points = numpy.zeros(shape=(len(good_matches), 2))
        second_points = numpy.zeros_like(first_points)
        for i, match in enumerate(good_matches):
            # The match objects only provide which index in each keypoint list is a match. Use these to look up the
            # position information for the given keypoint.
            matched_keypoint1 = keypoints1[match.queryIdx]
            matched_keypoint2 = keypoints2[match.trainIdx]
            first_points[i][0] = matched_keypoint1.pt[0]
            first_points[i][1] = matched_keypoint1.pt[1]
            second_points[i][0] = matched_keypoint2.pt[0]
            second_points[i][1] = matched_keypoint2.pt[1]
        return (first_points, second_points)

    # def calculateTransform(self, image1: numpy.ndarray, image2: numpy.ndarray) -> numpy.ndarray:
    #     """!
    #     @brief Estimate the transform a camera experienced when moving between the two images.

    #     This method uses key points to find the transformation between images. It uses several helper functions and
    #     classes to find matching key points in each image, then computes a likely transform from them.

    #     @param image1 The image taken at the start of the transformation.
    #     @param image2 The image taken at the end of the transformation.
    #     @return numpy.ndarray A 3x3 numpy array representing the homogenous matrix for the 2D transformation.
    #     """
    #     detector = PointDetector(500)
    #     # self.keypoints1 = detector.detect(image1)
    #     # self.keypoints2 = detector.detect(image2)
    #     self._tempDetect(image1, image2)
    #     self._tempDescribe(image1, image2)
    #     result = self._matchAndCalculate()
    #     return result

    # def _createKeypoints(self, keypoint_array: numpy.ndarray) -> cv2.KeyPoint:
    #     """!
    #     @brief A helper function to create OpenCV keypoints from
    #     """

    # def _matchAndCalculate(self) -> numpy.ndarray:
    #     """!
    #     @brief Match keypoints between images and use those to determine the transformation between images.

    #     This method performs the calculations on the backend to compute the transformation between images. It leverages
    #     OpenCV for convenience. It first finds related keypoints between images using a FLANN based KDTree search. It
    #     then computes a viable transformation between images using those correspondences. This portion of the code is
    #     taken from https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html.

    #     @return numpy.ndarray Returns a 3x3 matrix representing a homogenous 2D tranformation matrix.
    #     """
    #     index_params = dict(algorithm=1, trees=5)
    #     search_params = dict(checks=50)
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     matches = flann.knnMatch(self.descriptions1, self.descriptions2, k=2)
    #     good = []
    #     for m, n in matches:
    #         if m.distance < 0.7*n.distance:
    #             good.append(m)
    #     src_pts = numpy.float32(
    #         [self.keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     dst_pts = numpy.float32(
    #         [self.keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #     print(M.shape)
    #     print(M)

    #     return numpy.identity(3)

    # def _tempDetect(self, image1, image2):
    #     sift = cv2.SIFT_create()
    #     self.keypoints1 = sift.detect(image1)
    #     self.keypoints2 = sift.detect(image2)

    # def _tempDescribe(self, image1, image2):
    #     sift = cv2.SIFT_create()
    #     (_, self.descriptions1) = sift.compute(image1, self.keypoints1)
    #     (_, self.descriptions2) = sift.compute(image2, self.keypoints2)
