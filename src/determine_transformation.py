import argparse
import cv2
import numpy


def processImage(filename: str, detector: cv2.Feature2D, descriptor: cv2.Feature2D) -> tuple[list[cv2.KeyPoint], numpy.ndarray]:
    """!
    @brief Load an image, detect keypoints within it, and describe those keypoints.

    @param filename The file to load the image from.
    @param detector The keypoint detector to use.
    @param descriptor The keypoint describer to use.
    @return tuple A tuple containing the list of keypoints and list of descriptions.
    """
    try:
        # Load the image first
        color_image = cv2.imread(filename)
        # Convert to grayscale
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # Detect keypoints
        keypoints = detector.detect(image)
        # Describe the keypoints
        (_, descriptions) = descriptor.compute(image, keypoints)
    except:
        # I don't really care about particular errors, just if the image can go through the whole process or not.
        raise ValueError(
            'Unable to process image for file {0:s}'.format(filename))
    return (keypoints, descriptions)


if __name__ == '__main__':
    # Use command line arguments to specify which images to read.
    parser = argparse.ArgumentParser(
        description='Determine the 2D transformation that likely occurred between two images')
    parser.add_argument('first_image', metavar='1', type=str,
                        help='The first image in the sequence')
    parser.add_argument('second_image', metavar='2', type=str,
                        help='The second image in the sequence')
    args = parser.parse_args()
    # Create the selected detector and describer
    detector = cv2.SIFT_create()
    # Get the keypoints and descriptions from each specified image file.
    (first_keypoints, first_descriptions) = processImage(
        args.first_image, detector, detector)

    # Try to load each image. If there is a problem with any of them, warn the user and exit.
    # try:
    #     first_image = loadImage(args.first_image)
    # except ValueError as ex:
    #     print('Unable to load {0:s}'.format(args.first_image))
    #     exit(-1)
    # try:
    #     second_image = loadImage(args.second_image)
    # except ValueError:
    #     print('Unable to load {0:s}'.format(args.second_image))
    #     exit(-1)
    # # Next, perform keypoint detection with the selected method.
    # temp_detector = cv2.SIFT_create()
    # first_keypoints = temp_detector.detect(first_image)
    # second_keypoints = temp_detector.detect(second_image)
    # print(len(first_keypoints))
    # print(len(second_keypoints))
    # # Then, describe each keypoint
    # (_, first_descriptors) = temp_detector.compute(first_image, first_keypoints)
    # (_, second_descriptors) = temp_detector.compute(
    #     second_image, second_keypoints)
    # # Then, determine the correspondences
    # index_params = dict(algorithm=1, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(first_descriptors, second_descriptors, k=2)
    # for i in range(len(matches)):
    #     if len(matches[i]) != 2:
    #         print('Other things!')
    #     print("{0}: {1} - {2}".format(i, type(matches[i][0]), matches[i][1]))
    # print(len(matches[0]))
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)
    # src_pts = numpy.float32(
    #     [first_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts = numpy.float32(
    #     [second_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # print(src_pts.shape)
    # print(dst_pts.shape)
