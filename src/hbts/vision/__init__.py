"""Perception layer built on the MediaPipe Tasks API.

The legacy ``mediapipe.solutions`` API used by the first prototype has been
removed from recent MediaPipe releases, so every analyzer here uses the
Tasks API (``FaceLandmarker``, ``GestureRecognizer``, ``PoseLandmarker``) in
``VIDEO`` running mode, which enables temporal tracking between frames.
"""
