# Definition of Done

- [x] Reprojection error reported, with calibration image count: 23 portrait images, 0.7036765220692852 pixels in the Colab run, 23 of 23 with corners found. Local check on the same copies: 0.70 pixels.
- [x] Test split committed before training started. First commit 9d7a791 on dev-test01, with docs and dataset/splits.txt, before any code. Later commits: dataset and calibration, then code, then trained model and results.
- [x] One command runs on a new image and prints width and height in mm plus confidence, and handles object not found and reference not found. Command: `python inference/measure.py --image <photo.jpg> --output <overlay.jpg>`, default size 512. If the object or reference is not found, it prints OBJECT_NOT_FOUND or REFERENCE_NOT_FOUND and exits with code 2, with no measurement.
- [x] Accuracy table for 10 instances, with raw outputs saved. See measurement/accuracy_table.md, measurement/results_512_undistorted.json, and measurement/results_512_raw.json.
- [x] Undistortion comparison reported. Undistorted mean absolute error 3.76 mm, raw mean absolute error 4.51 mm, on the same 10 photos.
- [x] Target met or missed, with reason written down. MO-1 met (3.76 mm, 2.05 percent, target 5 mm and 3 percent). MO-2 met (mean IoU 0.9841, target 0.90). mAP@0.5 and mAP@0.5:0.95 were not computed, so that part of item 3 is not met.
- [x] Everything not finished listed, with why. Item 7: no speed or accuracy target was set and measured, no timing runs were done. Item 8: no formal breaking condition test was run; the touching and 256 pixel failures are recorded as observed failures only. Item 9: refusal behaviour exists in code, but thresholds were not tuned and the cost in refused images was not counted. Item 3, mAP metrics: not computed by the training script.
- [x] Loom 5 minutes or less. Link: to be added. Link: https://www.loom.com/share/a2992687ec6b49039a6c937a79a0bcbf (4:36)
- [ ] Pull request link: to be added.
