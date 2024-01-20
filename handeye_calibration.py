import numpy as np
import cv2 as cv
import glob
import os

class HandEyeCalibration:
    def __init__(self, save_dir, img_dir, posefile, eye_to_hand=True, chessboardSize=(9,6), frameSize=(480,640), size_of_chessboard_squares_m=0.04):
        self.save_dir = save_dir
        # images
        self.img_dir = img_dir
        self.images = sorted(glob.glob(self.img_dir+'/*.png'), key=lambda x: int(os.path.split(x)[-1].split('.')[0][3:])) # images must be sorted to align with pose order
        # print(os.path.split(glob.glob(self.img_dir+'/*.png')[0])[-1].split('.')[0][3:])
        # chessboardSize and camera parameters
        self.chessboardSize = chessboardSize
        self.frameSize = frameSize
        self.size_of_chessboard_squares_m = size_of_chessboard_squares_m

        # intrinsic parameters and robot poses
        self.gripper2base = np.load(posefile)
        self.cameraMatrix = np.load(self.save_dir+"/camera_intrinsic/cameraMatrix.npy")
        self.dist = np.load(self.save_dir+"/camera_intrinsic/dist.npy")

        # eye to hand or eye in hand
        self.eye_to_hand = eye_to_hand

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # translation and rotation vectors
        # target2cam
        self.R_tar2cam = []
        self.t_tar2cam = []

        # gripper2base
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.H_gripper2base = []
        
        # invalid images index
        self.invalid_images = []

    def get_target2cam(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        objp = objp * self.size_of_chessboard_squares_m

        print("-----hand eye images are parsed by following order:-----")

        # calculate corners
        for i, image in enumerate(self.images):
            print(image)
            img = cv.imread(image)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), self.criteria)
                # Find the rotation and translation vectors. 
                # rvecs=[rx, ry, rz] in rad, tvecs=[x, y, z] in m (defined by size_of_chessboard_squares_m)
                ret, rvecs, tvecs = cv.solvePnP(objp, corners2, self.cameraMatrix, self.dist)
                # target2cam
                self.R_tar2cam.append(cv.Rodrigues(rvecs)[0]) # rotation vector to rotation matrix
                self.t_tar2cam.append(tvecs)
            else:
                print('no chessboard corners found:'+image)
                self.invalid_images.append(i)
        
    def get_gripper2base(self):
        # convert 6D pose to rotation matrix and tvec
        # pose [x,y,z,rx,ry,rz] in m and rad
        for i, pose in enumerate(self.gripper2base):
            # skip invalid images
            if i in self.invalid_images:
                continue

            r = cv.Rodrigues(pose[3:])[0] # rotation vector to rotation matrix
            t = pose[:3]

            if self.eye_to_hand:
                # inverse to base2gripper https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
                r = r.T
                t = -r.dot(t)

            self.R_gripper2base.append(r)
            self.t_gripper2base.append(t)

    def handeye_calibrate(self, method=cv.CALIB_HAND_EYE_TSAI):
        # return cam2base if eye to hand
        # return cam2gripper if eye in hand
        R, t = cv.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_tar2cam, self.t_tar2cam, method=cv.CALIB_HAND_EYE_TSAI)
        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))
        print('-------cam2base/cam2gripper TSAI---------------')
        print(H)

        R, t = cv.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_tar2cam, self.t_tar2cam, method=cv.CALIB_HAND_EYE_PARK)
        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))
        print('-------cam2base/cam2gripper PARK---------------')
        print(H)

        R, t = cv.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_tar2cam, self.t_tar2cam, method=cv.CALIB_HAND_EYE_HORAUD)
        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))
        print('-------cam2base/cam2gripper HORAUD-------------')
        print(H)

        R, t = cv.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_tar2cam, self.t_tar2cam, method=cv.CALIB_HAND_EYE_ANDREFF)
        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))
        print('-------cam2base/cam2gripper ANDREFF------------')
        print(H)

        R, t = cv.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_tar2cam, self.t_tar2cam, method=cv.CALIB_HAND_EYE_DANIILIDIS)
        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))
        print('-------cam2base/cam2gripper DANIILIDIS---------')
        print(H)
    
    def calibrate(self):
        self.get_target2cam()
        self.get_gripper2base()
        self.handeye_calibrate()

    def save(self, method=cv.CALIB_HAND_EYE_TSAI):
        R, t = cv.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_tar2cam, self.t_tar2cam, method=method)
        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))

        if self.eye_to_hand:   
            np.save(self.save_dir+"/H_cam2base.npy", np.array(H))
            print('-------H_cam2base saved---------')
        else:
            np.save(self.save_dir+"/H_cam2gripper.npy", np.array(H))
            print('-------H_cam2gripper saved---------')
