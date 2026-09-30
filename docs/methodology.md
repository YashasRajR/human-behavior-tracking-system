# Methodology

This document describes how each metric is computed and why. All angles are in
degrees; "image-left/right" refer to the displayed (mirrored) frame, which in
the default selfie view matches the user's own left and right.

## 1. Eye Aspect Ratio and blinks

For six contour points of an eye (outer corner `p1`, upper lid `p2, p3`,
inner corner `p4`, lower lid `p5, p6`):

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)
```

EAR is invariant to scale and in-plane rotation and drops sharply when the
eye closes (Soukupova and Cech, 2016).

The first prototype used one eye and a fixed threshold of 0.15. The current
detector:

| Aspect | Prototype | Current |
| --- | --- | --- |
| Signal | left eye only | mean of both eyes |
| Threshold | fixed 0.15 | `0.72 x` personal open-eye baseline |
| State change | single threshold | hysteresis: close below 0.72, reopen above 0.82 of baseline |
| What counts | any dip | closures lasting 50-500 ms |
| Blink rate | total / session time | sliding 60 s window |

Closures longer than 500 ms are counted separately as long closures; closures
longer than 1 s raise a *microsleep* event.

## 2. PERCLOS

PERCLOS is the proportion of time the eyes are closed within a sliding window
(default 60 s). It is the fatigue indicator most strongly associated with
lapses in vigilance (Wierwille et al., 1994; Dinges and Grace, 1998). The value
is reported only after 15 s of history. The fatigue label is:

* **Drowsy** - microsleep in progress, or PERCLOS at least twice the threshold
* **Tired** - PERCLOS above the threshold (default 15%), or three or more yawns in five minutes
* **Alert** - otherwise

## 3. Yawns

Mouth Aspect Ratio (MAR) is the mean of three inner-lip openings divided by
mouth width. A yawn is `MAR >= 0.55` held for at least 1.2 s; speech produces
short MAR peaks that do not satisfy the duration condition.

## 4. Head pose

Six landmarks (nose tip, chin, outer eye corners, mouth corners) are matched
to a generic 3-D face model and solved with Perspective-n-Point
(`cv2.solvePnP`). Focal length is approximated by the image width. The solver
is seeded with a frontal pose at a depth estimated from the inter-ocular
distance, which prevents convergence to the mirrored (roll close to 180)
solution. The prototype multiplied the output of `RQDecomp3x3` by 360 and used
pixel coordinates as the 3-D model, which does not produce angles in degrees.

Sign conventions: yaw > 0 means the nose points to image-right, pitch > 0
means up, roll > 0 means clockwise. These are verified by unit tests that
project a synthetic head with known rotation.

## 5. Gaze

For each eye, the iris centre is projected onto the eye's own corner-to-corner
axis and lid axis, giving a horizontal and a vertical ratio in `[0, 1]`.
Projecting onto the eye's axes rather than raw image x/y makes the ratio
robust to head roll.

## 6. Attention score

The score fuses three factors in `[0, 1]`, using deviations from the
calibrated neutral pose (`d_yaw`, `d_pitch`, `d_gaze_h`, `d_gaze_v`):

```
head  = exp(-ln2 * ((d_yaw / 22)^2 + (d_pitch / 18)^2))

gaze_yaw   = d_yaw   + K * d_gaze_h
gaze_pitch = d_pitch - K * d_gaze_v
gaze  = exp(-ln2 * ((gaze_yaw / 22)^2 + (gaze_pitch / 18)^2))

eyes  = 1 while eyes are open, 0 during closures longer than 0.4 s (smoothed)

score = 100 * (0.45 * head + 0.35 * gaze + 0.20 * eyes)      then EMA, tau = 0.8 s
```

The Gaussian falloff halves each factor at the tolerance angle. The key idea
is the *gaze* term: when people turn their head slightly while reading, the
eyes counter-rotate to stay on the screen. Adding the eye-in-head rotation
(`K` degrees per unit of iris ratio, by default `22 / 0.16`) to the head angle
recognises that the user is still looking at the screen, where a head-pose-only
score would report a distraction. The user counts as *on screen* while
`gaze >= 0.5`.

All tolerances and weights are configurable (`attention.*`).

## 7. Calibration

During the first 4 s (and on demand with `C`) the engine collects:

* open-eye EAR baseline: **75th percentile** (blinks during calibration pull a mean down)
* neutral yaw, pitch, gaze ratios, neck angle and head-drop ratio: **median**

Calibration waits while no face is visible, so a user who sits down late does
not get an empty baseline. Event rules are suspended while calibrating and
calibration samples are excluded from report statistics.

## 8. Posture

From the pose landmarks, visible from a desk webcam:

* **Shoulder tilt**: angle of the shoulder line
* **Neck angle**: lateral inclination of the ear midpoint over the shoulder midpoint
* **Head drop**: vertical nose-to-shoulder distance divided by shoulder width

Slouching or leaning towards the screen lowers the head relative to the
shoulders; posture is flagged when head drop falls below 80% of the calibrated
value, the neck angle deviates by more than 18 degrees, or the shoulders tilt by
more than 9 degrees.

## 9. Events

| Event | Condition | Onset | Severity |
| --- | --- | --- | --- |
| `absent` | no face | 2 s | warning |
| `distracted` | face visible, gaze off screen, eyes open | 3 s | warning |
| `low_attention` | attention < 45 | 10 s | warning |
| `microsleep` | eyes closed | 1 s | critical |
| `drowsy` | PERCLOS >= 2 x threshold | 5 s | critical |
| `poor_posture` | posture flagged | 8 s | warning |
| `yawn`, `bookmark`, `snapshot`, `calibrated` | instant | - | info |

Event start times are back-dated to when the condition first became true, so
recorded durations are not shortened by the onset delay.

## 10. Gesture commands

A bound gesture must be held for 1.2 s (dwell time) with confidence of at
least 0.6. After firing, the gesture must be released before it can fire
again, and a 2 s cool-down applies. The dashboard draws a filling ring around
the hand during the hold so the user sees the command coming and can abort by
relaxing the hand.

## 11. Signal smoothing

Head angles and gaze ratios pass through a One Euro filter (Casiez et al.,
2012), which smooths strongly when the signal is still and weakly when it
moves fast, avoiding the lag of a plain moving average. EAR is deliberately
*not* smoothed, because smoothing would attenuate blinks.

## Limitations

* Gaze is estimated from iris position relative to the eye, not a calibrated
  screen-point gaze tracker; it indicates direction, not the exact point of
  regard.
* The head model is generic; absolute angles can be biased by a few degrees
  for individual faces. Calibration removes the constant part of this bias.
* Blink and PERCLOS values are behavioural indicators and are not a medical
  or safety-certified assessment.

## References

1. T. Soukupova and J. Cech. *Real-Time Eye Blink Detection using Facial Landmarks.* CVWW, 2016.
2. W. W. Wierwille, S. S. Wreggit, C. L. Kirn, L. A. Ellsworth, R. J. Fairbanks. *Research on Vehicle-Based Driver Status/Performance Monitoring.* NHTSA, 1994.
3. D. F. Dinges and R. Grace. *PERCLOS: A Valid Psychophysiological Measure of Alertness as Assessed by Psychomotor Vigilance.* FHWA-MCRT-98-006, 1998.
4. G. Casiez, N. Roussel, D. Vogel. *1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.* CHI, 2012.
5. C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172, 2019.
