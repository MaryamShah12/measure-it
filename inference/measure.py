"""One command: image to millimetres.

Steps, in order:
1. Undistort the image with the saved calibration, unless --no-undistort.
2. Segment the notebook with the trained U-Net.
3. Find the largest mask contour and its 4 corners.
4. Find the checkerboard in the same undistorted image.
   Pixels per mm comes from the distance between neighbouring
   inner corners divided by the measured square size, 20 mm.
   This is only valid if the notebook and the checkerboard are
   flat in the same plane, and the photo is face on.
5. Width and height in mm are the pixel side lengths divided
   by pixels per mm. Long side first, short side second.

If the object is not found, or the reference is not found,
no measurement is printed and the exit code is 2.

Run from the repo root:
    python inference/measure.py --image <photo.jpg> --output <overlay.jpg>
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch

import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def order_by_angle(pts):
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def side_lengths(pts):
    pts = np.array(pts, dtype=np.float32)
    lengths = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        lengths.append(float(np.linalg.norm(a - b)))
    lengths.sort()
    short = (lengths[0] + lengths[1]) / 2.0
    long = (lengths[2] + lengths[3]) / 2.0
    return long, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="models/best_model.pth")
    ap.add_argument("--calibration", default="calibration/calibration.npz")
    ap.add_argument("--output", default=None)
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-undistort", action="store_true")
    args = ap.parse_args()

    data = np.load(args.calibration)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    img = cv2.imread(args.image)
    if img is None:
        print("OBJECT_NOT_FOUND: could not read image")
        raise SystemExit(2)
    h, w = img.shape[:2]

    if args.no_undistort:
        undistorted = img
    else:
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    small = cv2.resize(undistorted, (args.size, args.size))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask_small = (probs > 0.5).astype(np.uint8)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    prob_full = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("OBJECT_NOT_FOUND")
        raise SystemExit(2)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.01 * w * h:
        print("OBJECT_NOT_FOUND: mask too small")
        raise SystemExit(2)

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        corners = approx.reshape(-1, 2)
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
    corners = order_by_angle(corners)
    long_px, short_px = side_lengths(corners)
    confidence = float(prob_full[mask > 0].mean()) if (mask > 0).any() else 0.0

    # Reference: checkerboard inner corners in the same undistorted image.
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    ret, cb_corners = cv2.findChessboardCorners(
        gray, (10, 8), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        print("REFERENCE_NOT_FOUND")
        raise SystemExit(2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cb_corners = cv2.cornerSubPix(gray, cb_corners, (11, 11), (-1, -1), criteria)
    grid = cb_corners.reshape(8, 10, 2)
    horiz = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vert = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    mean_px_per_square = float(np.mean(np.concatenate([horiz.ravel(), vert.ravel()])))
    px_per_mm = mean_px_per_square / args.square_mm

    width_mm = long_px / px_per_mm
    height_mm = short_px / px_per_mm

    result = {
        "image": args.image,
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "confidence": confidence,
        "pixels_per_mm": float(px_per_mm),
        "undistorted": not args.no_undistort,
    }
    print(json.dumps(result, indent=1))
    print(f"width: {width_mm:.1f} mm, height: {height_mm:.1f} mm, confidence: {confidence:.3f}")

    if args.output:
        overlay = undistorted.copy()
        cv2.polylines(overlay, [corners.astype(int)], True, (0, 0, 255), 4)
        text = f"{width_mm:.1f} x {height_mm:.1f} mm  conf {confidence:.2f}"
        cv2.putText(overlay, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(args.output, overlay)


if __name__ == "__main__":
    main()
