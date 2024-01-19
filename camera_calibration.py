import numpy as np
import cv2 as cv
import glob
import os

class IntrinsicCalibration:
    def __init__(self, save_dir, img_dir, chessboardSize=(9,6), frameSize=(480,640), size_of_chessboard_squares_m=0.04):
        self.save_dir = save_dir
        # images
        self.img_dir = img_dir
        self.images = sorted(glob.glob(self.img_dir+'/*.png'), key=os.path.getmtime) # sort by time

        # chessboardSize and camera parameters
        self.chessboardSize = chessboardSize
        self.frameSize = frameSize
        self.size_of_chessboard_squares_m = size_of_chessboard_squares_m

        # intrinsic parameters
        self.cameraMatrix = None
        self.dist = None

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def calibrate(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        objp = objp * self.size_of_chessboard_squares_m

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # calculate corners
        for image in self.images:
            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, self.chessboardSize, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(1)

                # uncomment to check the chessboard corners by pressing 's'
                # k = cv.waitKey(0) & 0xFF
                # if k == ord('s'):
                #     continue
            else:
                print('no chessboard corners found:'+image)
                continue
        cv.destroyAllWindows()

        # calibration
        ret, self.cameraMatrix, self.dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print('-----intrinsic Matrix----')
        print(self.cameraMatrix)

        # Save the camera calibration result for later use
        # np.save("camera_1/rvecs.npy", np.array(rvecs))
        # np.save("camera_1/tvecs.npy", np.array(tvecs))
        np.save(self.save_dir+"/camera_intrinsic/cameraMatrix.npy", np.array(self.cameraMatrix))
        np.save(self.save_dir+"/camera_intrinsic/dist.npy", np.array(self.dist))

        print('-----intrinsic Matrix saved----')

    def draw_axis(self, img, corners, imgpts):
        corner = tuple(int(c) for c in corners[0].ravel())
        img = cv.line(img, corner, tuple(int(c) for c in imgpts[0].ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(int(c) for c in imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(int(c) for c in imgpts[2].ravel()), (0,0,255), 5)
        return img

    # draw world frame of the chessboard
    def draw_pose(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        for image in self.images:
            img = cv.imread(image)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), self.criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv.solvePnP(objp, corners2, self.cameraMatrix, self.dist)
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, self.cameraMatrix, self.dist)
                img = self.draw_axis(img,corners2,imgpts)
                cv.imshow('img',img)
                # cv.waitKey(1)
                k = cv.waitKey(0) & 0xFF
                if k == ord('s'):
                    continue
                #     cv.imwrite(fname[:6]+'.png', img)
            else:
                print('no chessboard corners found:'+image)
                continue
        cv.destroyAllWindows()
