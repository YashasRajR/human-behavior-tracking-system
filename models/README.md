# Models

HBTS uses three MediaPipe Tasks model bundles. They are downloaded into this
folder automatically on first run (or ahead of time with `hbts models`) and are
excluded from version control.

| File | Used for | Source |
| --- | --- | --- |
| `face_landmarker.task` | 478-point face mesh with iris landmarks | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| `gesture_recognizer.task` | 21-point hand landmarks and 7 gesture classes | [MediaPipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) |
| `pose_landmarker_lite.task` | 33-point body pose | [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |

If your network blocks the download, fetch the files manually from the URLs in
`src/hbts/io/assets.py` and place them here with the names above.
