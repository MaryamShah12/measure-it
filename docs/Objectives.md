# Objectives

## MO-1: measurement accuracy (required)

Measure notebook width and height in mm from the one command, versus ruler ground truth, on 10 held out measurement photos that were not used for training.

Report: per instance table, mean absolute error, mean percentage error, median and worst case.

Target: mean absolute error 5 mm or less, and mean percentage error 3 percent or less.

Conditions: iPhone 14 Pro Max, 1x, face on, indoor light, notebook and checkerboard coplanar. Excluded: photos where the reference is partly hidden, the notebook is partly out of frame, or the camera is tilted.

Justification: 5 mm is about 2 percent on a notebook sized object.

## MO-2: mask quality

Mask IoU on the 7 image test split, model mask versus hand checked 4 corner polygon mask. IoU per image, then mean.

Target: mean IoU 0.90 or more. Same test split for every run. Undistorted images only.

## Reporting rules

Every number is pasted as raw output from the run that produced it. Docs state CPU, GPU, RAM, operating system, camera model, and image resolution. Timings are measured 5 times, with median and spread. Errors are reported per instance, then mean, median, and worst case.
