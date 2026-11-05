# Objectives

## MO-1: measurement accuracy (required)

Measure notebook width and height in mm from the one command, versus ruler ground truth, on 10 held out measurement photos that were not used for training.

Report: per instance table, mean absolute error, mean percentage error, median and worst case.

Target: mean absolute error 5 mm or less, and mean percentage error 3 percent or less.

Conditions: iPhone 14 Pro Max, 1x, face on, indoor light, notebook and checkerboard coplanar, with a clear gap between them. Excluded: photos where the reference is partly hidden, the notebook is partly out of frame, or the camera is tilted.

Justification: 5 mm is about 2 percent on a notebook sized object.

Result: met. On gap_001.jpg to gap_010.jpg, run at size 512 with undistortion, mean absolute error 3.76 mm, mean percentage error 2.05 percent, median absolute error 3.8 mm, worst case 7.2 mm. Full table in measurement/accuracy_table.md. Raw outputs in measurement/results_512_undistorted.json.

Undistortion comparison (item 6): same 10 photos with --no-undistort, size 512: mean absolute error 4.51 mm, mean percentage error 2.43 percent, median absolute error 4.7 mm, worst case 8.4 mm. Skipping undistortion made the mean absolute error 0.76 mm worse. Raw outputs in measurement/results_512_raw.json.

## MO-2: mask quality

Mask IoU on the 7 image test split, model mask versus hand checked 4 corner polygon mask. IoU per image, then mean.

Target: mean IoU 0.90 or more. Same test split for every run. Undistorted images only.

Result: met. Test mean IoU 0.9841. Per image: notebook_010.jpg 0.9841, notebook_020.jpg 0.978, notebook_030.jpg 0.9864, notebook_040.jpg 0.9891, notebook_050.jpg 0.9841, notebook_060.jpg 0.9836, notebook_070.jpg 0.9831. Precision 0.9861 to 0.9932, recall 0.9917 to 0.9963. Best validation IoU 0.9828 at epoch 30. mAP@0.5 and mAP@0.5:0.95 were not computed, so item 3 is only partly met on the metrics listed in the assessment document.

## Reporting rules

Every number is pasted as raw output from the run that produced it. Machine details as reported by the Colab run: device cuda, torch 2.11.0+cu130, segmentation-models-pytorch 0.6.0. The Colab GPU was a T4. Host CPU, RAM, and operating system were not recorded in the run output, so they are not claimed here. Camera model: iPhone 14 Pro Max. Image resolution used: 960 by 1280. Timings were not measured 5 times, so no timing claim is made. Errors are reported per instance, then mean, median, and worst case.
