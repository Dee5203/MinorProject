import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ---------- Optional audio (background noise) ----------
AUDIO_AVAILABLE = True
try:
    import sounddevice as sd
except Exception:
    AUDIO_AVAILABLE = False
    sd = None

# ===================== TUNABLE SETTINGS =====================

# Warning limits
MAX_FACE_PUPIL_GAZE_WARNINGS = 5   # shared pool (face OR pupil OR gaze)
MAX_NOISE_WARNINGS = 3             # separate pool (background noise)
COOLDOWN_SEC = 5                   # wait after each warning

# Face in-range margin (relative to face box)
FACE_MARGIN_X = 0.10
FACE_MARGIN_Y = 0.12

# -------- FIXED pupil tolerance --------
PUPIL_X_RANGE = (0.20, 0.80)   # looser than before
PUPIL_Y_RANGE = (0.25, 0.75)   # looser than before

SMOOTH_FRAMES = 7              # was 5
PUPIL_OUT_FRAMES_REQUIRED = 10 # must be out 10 frames (~0.3s) before warning

# -------- FIXED gaze tolerance --------
GAZE_CENTER_X_RANGE = (0.35, 0.65)   # wider center zone
GAZE_CENTER_Y_RANGE = (0.35, 0.65)   # wider center zone
GAZE_OUT_FRAMES_REQUIRED = 10        # must be out 10 frames before warning

# Mouth "speaking" threshold
SPEAKING_GAP_THRESHOLD = 0.030

# ===================== STATE =====================
warnings_face_pool = 0
warnings_noise = 0
last_facepool_warning_time = 0.0
last_noise_warning_time = 0.0
candidate_speaking = False
pupil_out_counter = 0
gaze_out_counter = 0

# Audio threshold
noise_threshold_rms = 0.03
SAMPLE_RATE = 16000

# ===================== MEDIAPIPE =====================
mp_face_mesh = mp.solutions.face_mesh

IDX_FOREHEAD = 10
IDX_CHIN = 152
IDX_LEFT_EAR = 234
IDX_RIGHT_EAR = 454
IDX_NOSE = 1
IDX_LIP_TOP = 13
IDX_LIP_BOTTOM = 14
IDX_PUPIL_L = 468
IDX_PUPIL_R = 473

left_hist = deque(maxlen=SMOOTH_FRAMES)
right_hist = deque(maxlen=SMOOTH_FRAMES)
gaze_hist = deque(maxlen=SMOOTH_FRAMES)

def smooth_point(hist, pt):
    hist.append(pt)
    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def smooth_scalar(hist, val):
    hist.append(val)
    return sum(hist) / len(hist)

# --------------------- Audio helpers ---------------------
def calibrate_noise_threshold():
    global noise_threshold_rms
    try:
        audio = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype='float32')
        sd.wait()
        rms = float(np.sqrt(np.mean(np.square(audio))))
        noise_threshold_rms = max(0.02, rms * 3.0)
        print(f"[Audio] Calibrated noise threshold RMS = {noise_threshold_rms:.4f}")
        return True
    except Exception as e:
        print(f"[Audio] Calibration failed: {e}. Disabling noise detection.")
        return False

def audio_callback(indata, frames, time_info, status):
    global warnings_noise, last_noise_warning_time
    if warnings_noise >= MAX_NOISE_WARNINGS:
        return
    rms = float(np.sqrt(np.mean(np.square(indata))))
    if (not candidate_speaking) and (rms > noise_threshold_rms):
        now = time.time()
        if now - last_noise_warning_time >= COOLDOWN_SEC:
            warnings_noise += 1
            last_noise_warning_time = now
            print(f"[Noise Warning] Loud background while silent ({warnings_noise}/{MAX_NOISE_WARNINGS})")

# --------------------- Geometry helpers ---------------------
def clamp01(x): return max(0.0, min(1.0, x))

def norm_to_face_box(px, py, lx, rx, fy, cy):
    nx = (px - lx) / max(1e-6, (rx - lx))
    ny = (py - fy) / max(1e-6, (cy - fy))
    return clamp01(nx), clamp01(ny)

def in_range(val, lo, hi): return (lo <= val <= hi)

# --------------------- Main ---------------------
cap = cv2.VideoCapture(0)

audio_stream = None
if AUDIO_AVAILABLE:
    try:
        if calibrate_noise_threshold():
            audio_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
            audio_stream.start()
        else:
            AUDIO_AVAILABLE = False
    except Exception as e:
        print(f"[Audio] Could not start input stream: {e}. Disabling noise detection.")
        AUDIO_AVAILABLE = False

with mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as mesh:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        line_color = (0, 255, 0)
        pupil_color_L = (0, 255, 0)
        pupil_color_R = (0, 255, 0)

        multi = results.multi_face_landmarks or []

        if len(multi) > 1:
            cv2.putText(frame, "Multiple faces detected!", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if len(multi) >= 1:
            lm = multi[0].landmark

            fx, fy = int(lm[IDX_FOREHEAD].x * w), int(lm[IDX_FOREHEAD].y * h)
            cx, cy = int(lm[IDX_CHIN].x * w), int(lm[IDX_CHIN].y * h)
            lx, ly = int(lm[IDX_LEFT_EAR].x * w), int(lm[IDX_LEFT_EAR].y * h)
            rx, ry = int(lm[IDX_RIGHT_EAR].x * w), int(lm[IDX_RIGHT_EAR].y * h)
            nx, ny = int(lm[IDX_NOSE].x * w), int(lm[IDX_NOSE].y * h)

            face_w = max(1, rx - lx)
            face_h = max(1, cy - fy)

            lip_gap = abs((lm[IDX_LIP_BOTTOM].y - lm[IDX_LIP_TOP].y))
            speaking_est = (lip_gap * h) / max(1.0, face_h)
            candidate_speaking = speaking_est > SPEAKING_GAP_THRESHOLD

            inset_l = lx + int(face_w * FACE_MARGIN_X)
            inset_r = rx - int(face_w * FACE_MARGIN_X)
            inset_t = fy + int(face_h * FACE_MARGIN_Y)
            inset_b = cy - int(face_h * FACE_MARGIN_Y)
            face_ok = (inset_l <= nx <= inset_r) and (inset_t <= ny <= inset_b)

            rawL = (lm[IDX_PUPIL_L].x * w, lm[IDX_PUPIL_L].y * h)
            rawR = (lm[IDX_PUPIL_R].x * w, lm[IDX_PUPIL_R].y * h)
            smL = smooth_point(left_hist, rawL)
            smR = smooth_point(right_hist, rawR)

            nLx, nLy = norm_to_face_box(smL[0], smL[1], lx, rx, fy, cy)
            nRx, nRy = norm_to_face_box(smR[0], smR[1], lx, rx, fy, cy)

            gx = (nLx + nRx) / 2.0
            gy = (nLy + nRy) / 2.0
            gx, gy = smooth_point(gaze_hist, (gx, gy))

            if gx < 0.35: gaze = "Looking Left"
            elif gx > 0.65: gaze = "Looking Right"
            elif gy < 0.35: gaze = "Looking Up"
            elif gy > 0.65: gaze = "Looking Down"
            else: gaze = "Looking Center"

            now = time.time()

            if not face_ok:
                if now - last_facepool_warning_time >= COOLDOWN_SEC:
                    warnings_face_pool += 1
                    last_facepool_warning_time = now
                    print(f"[Face Warning] {warnings_face_pool}/{MAX_FACE_PUPIL_GAZE_WARNINGS}")
                line_color = (0, 0, 255)
                pupil_color_L = (0, 0, 255)
                pupil_color_R = (0, 0, 255)
                pupil_out_counter = 0
                gaze_out_counter = 0
            else:
                line_color = (0, 255, 0)

                pupil_ok_L = in_range(nLx, *PUPIL_X_RANGE) and in_range(nLy, *PUPIL_Y_RANGE)
                pupil_ok_R = in_range(nRx, *PUPIL_X_RANGE) and in_range(nRy, *PUPIL_Y_RANGE)
                gaze_center = (in_range(gx, *GAZE_CENTER_X_RANGE) and
                               in_range(gy, *GAZE_CENTER_Y_RANGE))

                if not pupil_ok_L: pupil_color_L = (0, 0, 255)
                if not pupil_ok_R: pupil_color_R = (0, 0, 255)

                # Pupil violation counter
                if not pupil_ok_L or not pupil_ok_R:
                    pupil_out_counter += 1
                else:
                    pupil_out_counter = 0

                # Gaze violation counter
                if not gaze_center:
                    gaze_out_counter += 1
                else:
                    gaze_out_counter = 0

                if (pupil_out_counter >= PUPIL_OUT_FRAMES_REQUIRED or
                    gaze_out_counter >= GAZE_OUT_FRAMES_REQUIRED):
                    if now - last_facepool_warning_time >= COOLDOWN_SEC:
                        warnings_face_pool += 1
                        last_facepool_warning_time = now
                        print(f"[Pupil/Gaze Warning] {warnings_face_pool}/{MAX_FACE_PUPIL_GAZE_WARNINGS}")

            cv2.line(frame, (fx, fy), (cx, cy), line_color, 2)
            cv2.line(frame, (lx, ly), (rx, ry), line_color, 2)

            cv2.circle(frame, (int(smL[0]), int(smL[1])), 6, pupil_color_L, -1)
            cv2.circle(frame, (int(smR[0]), int(smR[1])), 6, pupil_color_R, -1)

            cv2.putText(frame, gaze, (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            if candidate_speaking:
                cv2.putText(frame, "Speaking", (w - 160, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Face/Pupil/Gaze: {warnings_face_pool}/{MAX_FACE_PUPIL_GAZE_WARNINGS}",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if AUDIO_AVAILABLE:
            cv2.putText(frame, f"Noise: {warnings_noise}/{MAX_NOISE_WARNINGS}",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(frame, "Noise: disabled", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

        eliminated = False
        elim_msg = None
        if warnings_face_pool >= MAX_FACE_PUPIL_GAZE_WARNINGS:
            eliminated = True
            elim_msg = "Eliminated: Too many face/pupil/gaze warnings"
        if warnings_noise >= MAX_NOISE_WARNINGS:
            eliminated = True
            if elim_msg is None:
                elim_msg = "Eliminated: Background noise (3 warnings)"

        if eliminated:
            cv2.putText(frame, elim_msg, (40, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.imshow("AI Proctoring", frame)
            cv2.waitKey(1500)
            break

        cv2.imshow("AI Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if audio_stream is not None:
    try:
        audio_stream.stop(); audio_stream.close()
    except Exception:
        pass
cv2.destroyAllWindows()
