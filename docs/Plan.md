# Plan

## Measure It: segment it, measure it, prove it

Object: blue UNICEF notebook, closed, blue cover up, flat.
Ground truth size: long side 204 mm, short side 158 mm, measured with a 30 cm ORO ruler, from the 0 line.

Camera: iPhone 14 Pro Max, 1x, JPG, Live Photo off, AE/AF locked, no zoom, phone upright (portrait) for every photo.
Image size used: 960 by 1280, WhatsApp transfer copies of photos I took. Originals are kept on the phone.

Background: white tiled floor. The notebook is the only object in training photos.

Reference, fallback: the printed checkerboard squares themselves, in measurement photos only, flat in the same plane as the notebook. Checkerboard: 11 by 9 squares, measured square size 20 mm (5 squares measured 10 cm exactly). The printed ArUco marker was not printed, so it is not used. No ruler and no IKOS device appear in any photo.

## Calibration

23 portrait checkerboard photos, all with 80 inner corners found. OpenCV calibrateCamera, square size 20 mm. Reprojection error on these copies: 0.70 pixels. Final calibration will be recomputed from the calibration images in this repo, and the final numbers will be pasted from the run output.

## Dataset

72 unique notebook photos, notebook_001.jpg to notebook_072.jpg. 17 exact duplicate files sent in error were removed before the split was fixed. No downloaded images.

Split, fixed before training, in dataset/splits.txt: train 51, val 14, test 7. Test: 010, 020, 030, 040, 050, 060, 070. Val: 004, 009, 014, 019, 024, 029, 034, 039, 044, 049, 054, 059, 064, 069. Train: the rest. Test images will never be used for training, tuning, or checkpoint selection.

Labelling: 4 corners per photo, filled polygon mask. Each polygon was checked against the blue cover outline. Masks are in dataset/masks.

## Model

U-Net with a pretrained ResNet18 encoder, one class, notebook versus background. Not YOLO and not Roboflow. Reason: small dataset, pretrained backbone, fast on Colab T4. ResNet18 was chosen over ResNet34 so the saved weights stay under the GitHub 100 MB file limit. The task is a simple blue shape on a light floor, so the smaller encoder is enough.

Training: Colab T4, seed fixed, config pinned in the training script when it is added.

## Inference, one command

Undistort the input image, segment the notebook, find the checkerboard reference in the same photo, convert pixels to mm using the measured 20 mm square size, and output a mask overlay with width in mm, height in mm, and confidence. If the object is not found, say so. If the reference is not found, say so. No measurement is printed in those cases.

## Measurement

10 new measurement photos, taken after training, face on, notebook plus checkerboard flat in the same plane, not part of the 72 training photos. Ground truth for each photo is the ruler measurement, 204 mm by 158 mm. Per instance table, mean absolute error and mean percentage error will be reported. Raw versus undistorted comparison will be attempted after the main measurement, only if time allows.

## Scope

Aim: items 1 to 4, plus item 5. Attempt item 6 if time allows. Items 8 and 9 are skipped unless time remains, and that will be stated honestly in the final record.

## Decision record

To be added at the end, with what was changed, why, and what it cost.
