# cv_get_object_position

The `cv_get_object_position` is to get object 3d position on map frame by yolov5-ros.
In addition, this is ROS package.

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).


**Content:**
* [Setup](#setup)
* [Execution of program](#execution-of-program)
* [Files](#files)
* [References](#References)


## Setup
~~~
cd /cv_get_object_position/cv_get_object_position/bash
bash reset_data_folder.bash
~~~

~~~
catkin_make (or catkin build)
~~~

Please input target labels and number in `target_objects.yaml` of `config`.



## Execution of program
~~~
cd /cv_get_object_position/cv_get_object_position/src/cv_get_object_position
python cv_get_object_position.py
~~~


## Files
 - `README.md`: Read me file (This file)

 - `__init__.py`: Set initial parameters

 - `cv_get_object_position.py`: main program


## Reference
yolov5-ros  
https://github.com/Nenetti/yolov5-ros  
