You can use the image data to do your own calibration and reconstruction, or use the calibrated result to do your reconstruction.

1. In groundtruth data folder, the scanned result use the PLY format.

2. In image data folder, there are all the images we captured.

3. In projection matrix folder, there are two files. Here we donot use the markers to calibrate the cameras, but use the selfcalibration method. So the result compared with the groundtruth need a similar transformation. You can use our benchmark data evaluation algorithm to find the similar transformation.

Because we use the selfcalibration method to calibrate the cameras, so not all the images are calibrated. The names of the calibrated images are stored in ImageList.txt. And the corresponding projection matrixes are stored in ProjectionMatrix.txt. We reshape the 3*4 projection matrix to 1*12 vector in row wise. The two distortion factors of the camera are stored in the end of the vector. So the vector of the camera is 1*14.  