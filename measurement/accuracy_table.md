# Accuracy table: 10 measurement photos, size 512

Ground truth for every photo: 204 mm by 158 mm, measured with a 30 cm ORO ruler.
Undistorted run: `python inference/measure.py --image <photo> --size 512 --output <overlay>`
Raw run: same command plus `--no-undistort`.

| Photo | Undistorted width (mm) | Undistorted height (mm) | Width error (mm) | Height error (mm) | Raw width (mm) | Raw height (mm) | Raw width error (mm) | Raw height error (mm) | Confidence (undistorted) |
|---|---|---|---|---|---|---|---|---|---|
| gap_001.jpg | 199.0 | 157.4 | -5.0 | -0.6 | 196.8 | 151.4 | -7.2 | -6.6 | 0.959 |
| gap_002.jpg | 198.6 | 157.1 | -5.4 | -0.9 | 196.5 | 152.3 | -7.5 | -5.7 | 0.962 |
| gap_003.jpg | 197.4 | 156.7 | -6.6 | -1.3 | 195.9 | 155.2 | -8.1 | -2.8 | 0.976 |
| gap_004.jpg | 198.5 | 154.9 | -5.5 | -3.1 | 195.6 | 158.0 | -8.4 | 0.0 | 0.976 |
| gap_005.jpg | 196.8 | 154.0 | -7.2 | -4.0 | 197.9 | 153.7 | -6.1 | -4.3 | 0.978 |
| gap_006.jpg | 203.6 | 163.6 | -0.4 | 5.6 | 202.9 | 158.1 | -1.1 | 0.1 | 0.971 |
| gap_007.jpg | 201.7 | 161.6 | -2.3 | 3.6 | 198.9 | 159.9 | -5.1 | 1.9 | 0.967 |
| gap_008.jpg | 203.2 | 163.9 | -0.8 | 5.9 | 200.8 | 160.2 | -3.2 | 2.2 | 0.961 |
| gap_009.jpg | 196.8 | 155.4 | -7.2 | -2.6 | 197.3 | 153.9 | -6.7 | -4.1 | 0.978 |
| gap_010.jpg | 199.2 | 155.7 | -4.8 | -2.3 | 197.3 | 155.5 | -6.7 | -2.5 | 0.980 |

Undistorted: mean absolute error 3.76 mm, mean percentage error 2.05 percent, median absolute error 3.8 mm, worst case 7.2 mm.
Raw (no undistortion): mean absolute error 4.51 mm, mean percentage error 2.43 percent, median absolute error 4.7 mm, worst case 8.4 mm.
Skipping undistortion made the mean absolute error 0.76 mm worse.

Raw outputs are saved in measurement/results_512_undistorted.json and measurement/results_512_raw.json. These numbers are pasted from the Colab run output on 2026-11-04.
