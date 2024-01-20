# handeye_calibration

## Installation
    ```git clone git@github.com:xiaoBOSS97/handeye_calibration.git```
        
    ```cd handeye_calibration```
    
    ```pip install -r requirements.txt```

## Usage

### step 1. sample calibration images and robot arm poses

- store camera calibration images in `/eye2hand_cam/img_1` as png files, around 20-30 images 

- store hand eye calibration images in `/eye2hand_cam/img_2` as png files, around 20-30 images (better different from camera calibration images)

- store the corresponding robot arm poses in `/eye2hand_cam/img_2/pose.npy` as npy files, example format [x, y, z, rx, ry, rz]

### step 2. run calibration
    ```python main.py```

## Watch out!
- the images readed by glob.glob() are in random order, so you need to check the order of the images and poses in the `pose.npy` file, and make sure the order is correct.
