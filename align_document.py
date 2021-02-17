from alignment.align_images import align_images
import numpy as np
import argparse
import imutils
import cv2

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True, help="path to input template image")
args = vars(ap.parse_args())

# load input image and template from storage
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# align the images
print("[INFO] aligning images...")
aligned = align_images(image, template, debug=True)

# resize images and tempolate to visualise
aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

# side-by-side comparison
stacked = np.hstack([aligned, template])

# ovelay second image
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

# showing two output
cv2.imshow("Image alignment stack", stacked)
cv2.imshow("Image alignment overlay", output)
cv2.waitKey(0)