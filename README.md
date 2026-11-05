# Measure It

Segment a blue UNICEF notebook, measure it in mm, and prove the accuracy against a ruler.

Camera: iPhone 14 Pro Max, 1x, JPG, portrait, 960 by 1280 images used.
Object: blue UNICEF notebook, 204 mm by 158 mm.
Reference: printed checkerboard squares, 20 mm per square, in measurement photos only, flat in the same plane as the notebook, with a clear gap between them.

Layout: calibration/, dataset/, models/, inference/, measurement/, docs/, README.md, requirements.txt.

## Results

Calibration: 23 portrait checkerboard photos, reprojection error 0.7036765220692852 pixels.
Mask quality: test mean IoU 0.9841 on 7 test images, best validation IoU 0.9828.
Measurement: 10 photos, mean absolute error 3.76 mm, mean percentage error 2.05 percent, median 3.8 mm, worst case 7.2 mm. Without undistortion, on the same 10 photos, mean absolute error 4.51 mm.
Weights: models/best_model.pth, 54.8 MB as shown on GitHub. mAP@0.5 and mAP@0.5:0.95 were not computed. Items 7, 8, and 9 were not completed as formal tests. See docs/Definition_of_Done.md.

## One command

From the repo root:

```
python inference/measure.py --image <photo.jpg> --output <overlay.jpg>
```

The command undistorts the photo, segments the notebook, finds the checkerboard, and prints width in mm, height in mm, and confidence. If the object or reference is not found, it prints OBJECT_NOT_FOUND or REFERENCE_NOT_FOUND and exits with code 2, with no measurement. Inference size defaults to 512. The conversion is only valid if the notebook and checkerboard are flat in the same plane and the photo is face on.

## Status

Plan, Objectives, Definition of Done, and the fixed split were committed first, before any code. Later commits added the dataset and calibration, the code, the trained model and results, and the measurement photos and final results.
