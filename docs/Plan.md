# Plan

## Measure It: segment it, measure it, prove it

Object: blue UNICEF notebook, closed, blue cover up, flat.
Ground truth size: long side 204 mm, short side 158 mm, measured with a 30 cm ORO ruler, from the 0 line.

Camera: iPhone 14 Pro Max, 1x, JPG, Live Photo off, AE/AF locked, no zoom, phone upright (portrait) for every photo.
Image size used: 960 by 1280, WhatsApp transfer copies of photos I took. Originals are kept on the phone.

Background: white tiled floor. The notebook is the only object in training photos.

Reference, fallback: the printed checkerboard squares themselves, in measurement photos only, flat in the same plane as the notebook. Checkerboard: 11 by 9 squares, measured square size 20 mm (5 squares measured 10 cm exactly). The printed ArUco marker was not printed, so it is not used. No ruler and no IKOS device appear in any photo.

## Calibration

23 portrait checkerboard photos, all with 80 inner corners found. OpenCV calibrateCamera, square size 20 mm. Reprojection error: 0.70 pixels on the WhatsApp sized copies in the local check, and 0.7036765220692852 pixels in the Colab run of calibration/calibrate.py on the same 23 images, 23 of 23 with corners found. Camera matrix and distortion coefficients are saved in calibration/calibration.npz and calibration/calibration.json.

## Dataset

72 unique notebook photos, notebook_001.jpg to notebook_072.jpg. 17 exact duplicate files sent in error were removed before the split was fixed. No downloaded images.

Split, fixed before training, in dataset/splits.txt: train 51, val 14, test 7. Test: 010, 020, 030, 040, 050, 060, 070. Val: 004, 009, 014, 019, 024, 029, 034, 039, 044, 049, 054, 059, 064, 069. Train: the rest. Test images were never used for training, tuning, or checkpoint selection.

Labelling: 4 corners per photo, filled polygon mask. Each polygon was checked against the blue cover outline. Masks are in dataset/masks.

## Model

U-Net with a pretrained ResNet18 encoder, one class, notebook versus background. Not YOLO and not Roboflow. Reason: small dataset, pretrained backbone, fast on Colab T4. ResNet18 was chosen over ResNet34 so the saved weights stay under the GitHub 100 MB file limit. The task is a simple blue shape on a light floor, so the smaller encoder is enough.

Training: Colab T4, seed 42, 30 epochs, batch 8, input 256 by 256, Dice plus BCE loss, Adam lr 1e-4. Best validation IoU 0.9828 at epoch 30. Test mean IoU 0.9841, per image 0.978 to 0.9891, precision 0.9861 to 0.9932, recall 0.9917 to 0.9963. mAP@0.5 and mAP@0.5:0.95 were not computed by the training script, so they are not reported. See Objectives.md.

## Inference, one command

Undistort the input image, segment the notebook, find the checkerboard reference in the same photo, convert pixels to mm using the measured 20 mm square size, and output a mask overlay with width in mm, height in mm, and confidence. If the object is not found, say so. If the reference is not found, say so. No measurement is printed in those cases.

Inference size: 512, set as the default in inference/measure.py. Training used 256. The model was run at 512 for the final measurement because at 256 the mask joined the notebook to the checkerboard in one photo. See the decision record. Run from the repo root: `python inference/measure.py --image <photo.jpg> --output <overlay.jpg>`

## Measurement

10 measurement photos, gap_001.jpg to gap_010.jpg in measurement/, taken after training, face on, notebook plus checkerboard flat in the same plane, with a clear gap of floor showing between them, not part of the 72 training photos. Ground truth for each photo is the ruler measurement, 204 mm by 158 mm. Per instance table, mean absolute error and mean percentage error are in measurement/accuracy_table.md. The same 10 photos were also run with --no-undistort for the undistortion comparison in item 6.

## Scope

Done: items 1 to 6. Items 7, 8, and 9 were not completed as formal tests, and that is stated honestly below and in Definition_of_Done.md. The refusal behaviour in measure.py (OBJECT_NOT_FOUND / REFERENCE_NOT_FOUND, exit code 2, no measurement) exists in code, but refusal thresholds were not tuned and the cost in refused images was not counted, so item 9 is not claimed as done. No speed target was set and measured, so item 7 is not done. No formal breaking condition test was run, so item 8 is not done, although the touching and 256 pixel failures are recorded as observed failure modes.

## Decision record

- ResNet34 to ResNet18: weights had to stay under the GitHub 100 MB limit. Cost: a smaller encoder. Result: test mean IoU 0.9841, weights 54.8 MB, so the cost did not hurt this task.
- 100 photos planned, 72 used: 17 duplicates were removed, leaving 72 unique photos. Cost: the test split is 7 images, not 10. The split was fixed before training, and the test images were never used for training, tuning, or checkpoint selection.
- ArUco marker not printed: the checkerboard squares themselves are the fallback reference, flat in the same plane. Cost: the reference is the checkerboard, not a separate marker, and the conversion is only valid face on with both flat in the same plane.
- WhatsApp sized copies, 960 by 1280, used for calibration, training, and measurement. Originals are kept on the phone. Cost: lower resolution than the originals, stated plainly.
- Measurement photos, first set, notebook touching the checkerboard: the model mask joined the two, and the red box ran into the checkerboard in 8 of 10 photos. Fix: retake with a clear gap. Cost: one extra photo session.
- Second measurement set, 6 photos, judged tilted: blue polygon estimates were 186.6 to 201.1 mm by 140.9 to 158.3 mm, well below truth. These photos were excluded under the Objectives exclusion for tilted photos, and were not used in the accuracy table.
- Inference size 256 to 512: at 256, gap_004 still joined the notebook to the checkerboard and gave 331.6 by 220.0 mm. At 512, all 10 photos measured correctly. Cost: slower inference, not timed, and the model itself was not retrained. The default in measure.py was changed to 512 so the one command reproduces the reported numbers.
- One thing I did not understand, for the recording: I had never done computer vision before this task. I did not know what camera calibration was, or why a photo had to be undistorted. I learned it by doing it one step at a time, checking the checkerboard corners, the RMS number, and the overlay boxes, and asking when an overlay box ran into the checkerboard.
