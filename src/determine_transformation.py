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
    (second_keypoints, second_descriptions) = processImage(
        args.second_image, detector, detector)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(
        queryDescriptors=first_descriptions, trainDescriptors=second_descriptions, k=2)
    good_threshold = 0.7
    good_matches = []
    for match_set in knn_matches:
        if match_set[0].distance < good_threshold * match_set[1].distance:
            good_matches.append(match_set[0])
    print('There are {0:d} good matches'.format(len(good_matches)))

    points1 = numpy.zeros(shape=(len(good_matches), 2))
    points2 = numpy.zeros_like(points1)
    for i, match in enumerate(good_matches):
        keypoint1 = first_keypoints[match.queryIdx]
        keypoint2 = first_keypoints[match.trainIdx]
        points1[i][0] = keypoint1.pt[0]
        points1[i][1] = keypoint1.pt[1]
        points2[i][0] = keypoint2.pt[0]
        points2[i][1] = keypoint2.pt[1]

    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(H)

    camera_parameters = numpy.array(
        [[1, 0, 0.1], [0, 1, 0.1], [0.0, 0.0, 1.0]])
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(
        H, camera_parameters)
    for i in range(num_solutions):
        print('Option #{0:d}:'.format(i))
        print('Rotation:')
        print(rotations[i])
        print('Translation:')
        print(translations[i])
        print('Plane Normal:')
        print(normals[i])
        print('\n')
