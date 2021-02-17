import numpy as np
import imutils
import cv2


def align_images(image, template, maxFeature=500, keepPercent=0.2, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # using ORB to detect keypoints
    orb = cv2.ORB_create(maxFeature)
    (kpsA, descA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descB) = orb.detectAndCompute(templateGray, None)

    # match features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descA, descB, None)

    # sorting the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # only top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # visualize the matches
    if debug:
        matchedVis = cv2.drawMatches(
            image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("matched keypoints", matchedVis)
        cv2.waitKey(0)

    # homographic matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over top matches
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute homographic matrix between two sets
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # align image
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned
