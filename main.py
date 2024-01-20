from camera_calibration import IntrinsicCalibration
from handeye_calibration import HandEyeCalibration
import cv2 as cv

if __name__== "__main__":
    chessboardSize=(9,6)
    frameSize=(480,640)
    size_of_chessboard_squares_m=0.04 # 4mm

    save_dir = '/home/zhi.zheng/zz/calibration/eye2hand_cam1'

    # calibrate camera
    cam_calib = IntrinsicCalibration(save_dir=save_dir, img_dir='/home/zhi.zheng/zz/calibration/eye2hand_cam1/img_2/')
    cam_calib.calibrate()
    # calib.draw_pose()

    # calibrate handeye
    handeye_calib = HandEyeCalibration(save_dir=save_dir, img_dir='/home/zhi.zheng/zz/calibration/eye2hand_cam1/img_4/', 
                                       posefile='/home/zhi.zheng/zz/calibration/eye2hand_cam1/img_4/pose.npy', 
                                       eye_to_hand=True)
    handeye_calib.calibrate()
    handeye_calib.save(method=cv.CALIB_HAND_EYE_TSAI)

