# python sfm

This is a python code for SFM, short for structure from motion, which can

reconstruct point cloud and localize the camera.

There is a sampled example dataset in data folder, you can run this example

by directly run "python sfm.py"

reconstructed result:

![image](https://github.com/ZouLG/py_sfm/tree/master/data/Figure.png)

matplotlib is rather slow when dealing with large amounts of points.

As a result, point cloud are saved to a ply file, you can visualize the 

ply file via meshlab or use open3d package, the script visualize.py in data 

is used for visualizing as following.

![image](https://github.com/ZouLG/py_sfm/tree/master/data/qinghuamen.jpg)
