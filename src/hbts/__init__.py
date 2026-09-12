"""HBTS: real-time, multi-modal human behavior tracking.

The package is organised in layers:

* ``hbts.core``      - pure maths: geometry, signal filters, shared data types
* ``hbts.vision``    - MediaPipe-backed perception (face, hands, pose)
* ``hbts.analytics`` - temporal reasoning on top of perception (blinks, fatigue,
  calibration, attention, events, gesture commands)
* ``hbts.io``        - video sources, model assets, session recording, reports
* ``hbts.ui``        - the OpenCV dashboard

Only ``hbts.vision`` and ``hbts.ui`` depend on MediaPipe / OpenCV at import
time; everything in ``core`` and ``analytics`` is plain NumPy and is unit
tested without a camera.
"""

__version__ = "2.0.0"
