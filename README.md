# image-stitching-homography
Create Image Stitching from obtained keypoints from using ORB (Orientated Brief and Rotated Fast), a computer vision technique.
Utilized RANSAC to find the best fitting Homography Matrix within a given iteration and applied it to an image to attempt to stitch it together.
Implemented homography and RANSAC by scratch - could maximize output by using cv2.findHomography() and cv2.warpPerspective()

## Tools
- OpenCV
- NumPy
- Python3

### TODO:
- Clean up stitching process and maximize best RANSAC results in Homography Matrix to reduce inconsistencies without using cv2.findHomography() and cv2.warpPerspective()
