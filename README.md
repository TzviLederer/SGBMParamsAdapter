# SGBMParametersAdjustment
This project if for SGBM (semi-global block matching) stereo matcher parameters adjustment. 
There are a lot of parameters to optimized, and in this project we put all the params on an intuitive GUI.

The parameters are described in:  
https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html

## How to use
In the `sgbm_parameters_finder.py` file, the class 'SGBMParameterFinder' is the main class. 
Create an instance of it:  
`sgbm_parameters_finder = SGBMParameterFinder(image_l, image_r)`  
the `image_l` and `image_r` has to be *rectified images*.

After it, do `sgbm_parameters_finder.play()` to play the GUI.

An example of the code is in `example.py`
