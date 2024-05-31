# What is Smart Fish Scanner?
The project is to establish an automated method of measuring fish length in order to streamline research.

## How To Use
### When using a single camera
Step 1. Take images for calibration
```python
python3 single_image.py
```
Step 2. Perform calibration
```python
python3 calibration_by_chessboard.py
```
Step 3. Take an image of the measurement target and correct the distortion.
```python
python3 undistort.py
```
Step 4. Measuring the length of the target from an image
```python
python3 measurement.py
```  
※Precautions for use
Measurement requires AR markers to be placed at the four corners,
By running create_marker.py, you can generate the AR markers you need. 
Place them in the order 0, 1, 2, 3, clockwise from the upper left.


### When using a stereo camera
Step 1.  Take images for calibration and perform calibration
```python
python3 stereo_calibration.py
```
Step 2. Take a image of the target and create a stereo image
```python
python3 depth.py
```
