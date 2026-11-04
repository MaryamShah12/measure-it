"""Calibrate the camera from checkerboard photos.

Finds the 10 by 8 inner corners in every image in calibration/images,
runs OpenCV calibrateCamera with a measured square size of 20 mm,
and saves the camera matrix, distortion coefficients and RMS error.

Run from the repo root:
    python calibration/calibrate.py
"""
import glob
import json

import cv2
import numpy as np

PATTERN = (10, 8)
SQUARE_MM = 20.0
IMAGE_SIZE = (960, 1280)  # width, height, portrait photos used here


def main():
    files = sorted(glob.glob("calibration/images/*.jpg"))
    if not files:
        raise SystemExit("No calibration images found in calibration/images")

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

    objpoints = []
    imgpoints = []
    found = 0
    for path in files:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret:
            print("no corners:", path)
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        found += 1

    print(f"images with corners: {found} of {len(files)}")
    if found < 10:
        raise SystemExit("Too few calibration images with corners found.")

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, IMAGE_SIZE, None, None
    )
    print("RMS reprojection error (pixels):", rms)
    print("camera_matrix:", camera_matrix.tolist())
    print("dist_coeffs:", dist_coeffs.ravel().tolist())

    np.savez(
        "calibration/calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        image_size=np.array(IMAGE_SIZE),
    )
    with open("calibration/calibration.json", "w") as f:
        json.dump(
            {
                "rms_px": float(rms),
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.ravel().tolist(),
                "image_size_wh": list(IMAGE_SIZE),
                "n_images": found,
                "square_mm": SQUARE_MM,
                "pattern_inner_corners": list(PATTERN),
            },
            f,
            indent=1,
        )

    # Save one undistorted example so the effect can be seen.
    img = cv2.imread(files[0])
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, IMAGE_SIZE, 1, IMAGE_SIZE
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_k)
    cv2.imwrite("calibration/undistorted_sample.jpg", undistorted)
    print("saved calibration/calibration.npz, calibration/calibration.json")


if __name__ == "__main__":
    main()
