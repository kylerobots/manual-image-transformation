import numpy
import cv2
from PointDetector import PointDetector


class TransformmationCalculator:
    """!
    @brief A class to calculate the estimated 2D transformation experienced by a camera, based on two images.
    """

    def calculateTransform(self, image1: numpy.ndarray, image2: numpy.ndarray) -> numpy.ndarray:
        """!
        @brief Estimate the transform a camera experienced when moving between the two images.

        This method uses key points to find the transformation between images. It uses several helper functions and
        classes to find matching key points in each image, then computes a likely transform from them.

        @param image1 The image taken at the start of the transformation.
        @param image2 The image taken at the end of the transformation.
        @return numpy.ndarray A 3x3 numpy array representing the homogenous matrix for the 2D transformation.
        """
        detector = PointDetector(500)
        # self.keypoints1 = detector.detect(image1)
        # self.keypoints2 = detector.detect(image2)
        self._tempDetect(image1, image2)
        self._tempDescribe(image1, image2)
        result = self._matchAndCalculate()
        return result

    def _createKeypoints(self, keypoint_array: numpy.ndarray) -> cv2.KeyPoint:
        """!
        @brief A helper function to create OpenCV keypoints from
        """

    def _matchAndCalculate(self) -> numpy.ndarray:
        """!
        @brief Match keypoints between images and use those to determine the transformation between images.

        This method performs the calculations on the backend to compute the transformation between images. It leverages
        OpenCV for convenience. It first finds related keypoints between images using a FLANN based KDTree search. It
        then computes a viable transformation between images using those correspondences. This portion of the code is
        taken from https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html.

        @return numpy.ndarray Returns a 3x3 matrix representing a homogenous 2D tranformation matrix.
        """
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.descriptions1, self.descriptions2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        src_pts = numpy.float32(
            [self.keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = numpy.float32(
            [self.keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(M.shape)
        print(M)

        return numpy.identity(3)

    def _tempDetect(self, image1, image2):
        sift = cv2.SIFT_create()
        self.keypoints1 = sift.detect(image1)
        self.keypoints2 = sift.detect(image2)

    def _tempDescribe(self, image1, image2):
        sift = cv2.SIFT_create()
        (_, self.descriptions1) = sift.compute(image1, self.keypoints1)
        (_, self.descriptions2) = sift.compute(image2, self.keypoints2)
