"""Echo AI kiosk loop.

State machine:

  IDLE   --(wake word)-->  ACTIVE
  ACTIVE --(30s no event)-> IDLE

In ACTIVE state, the right half of the screen is a tabbed panel:
  PROJECTS  (default)  - read-only list of projects from the DB.
  REGISTER             - in-window text fields to enrol a new face.
  CHAT                 - 5 questions per session, OpenAI w/ Ollama fallback.

In IDLE the camera keeps capturing (and the recogniser keeps running so we
can wake on faces if we ever want to), but the cv2 window shows the blue
welcome screen with weather and counter instead.

Press keys in the OpenCV window:
  q         quit
  P / R / C switch tab in ACTIVE
  V         (in CHAT) toggle voice/keyboard input
  Tab       (in REGISTER) switch between emp_id / name fields
  Enter     submit current field / submit chat question
  Esc       cancel current REGISTER / CHAT input
"""

from __future__ import annotations

# Suppress the harmless Qt font warning from opencv-python's bundled Qt
# before cv2 is imported.
import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

import argparse
import queue
import time
import uuid
from concurrent.futures import Future

import cv2
import numpy as np
from picamera2 import Picamera2

import config
import views
from async_tts import AsyncTTS
from blink import BlinkChecker
from chat import ChatBackendError, ChatBudgetError, ChatClient
from database import FaceDB
from hailo_infer import HailoFacePipeline, align_face
from liveness import LivenessChecker
from quality import (
    is_quality_face,
    landmark_anchor,
    landmarks_drift,
    shift_matches_direction,
)
from tts import make_backend
from wake_word import WakeWordError, WakeWordListener
from weather import WeatherPoller


# ---------- helpers --------------------------------------------------------


def cosine_match(query: np.ndarray, matrix: np.ndarray):
    if matrix.shape[0] == 0:
        return -1, 0.0
    sims = matrix @ query
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def best_self_match(embeddings, db_ids, db_matrix, emp_id) -> float:
    mask = np.array([eid == emp_id for eid in db_ids], dtype=bool)
    if not mask.any():
        return -1.0
    sub = db_matrix[mask]
    return max(float((sub @ e).max()) for e in embeddings)


def largest_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


# ---------- silent learning ------------------------------------------------


class SilentLearner:
    def __init__(self, db: FaceDB):
        self.db = db
        self._last_added: dict[str, float] = {}

    def maybe_add(self, emp_id: str, name: str,
                  embedding: np.ndarray, score: float) -> bool:
        if not config.SILENT_LEARN_ENABLED:
            return False
        if score < config.SILENT_LEARN_MIN_SCORE:
            return False
        now = time.time()
        if now - self._last_added.get(emp_id, 0) < config.SILENT_LEARN_MIN_INTERVAL_SEC:
            return False
        existing = self.db.get_embeddings(emp_id)
        if existing.shape[0] > 0:
            sims = existing @ embedding
            if float(sims.max()) > config.SILENT_LEARN_MAX_SIMILARITY:
                return False
        self.db.add_embedding(emp_id, embedding)
        self._last_added[emp_id] = now
        cap = config.SILENT_LEARN_MAX_SAMPLES_PER_PERSON
        n = self.db.count_embeddings(emp_id)
        if n > cap:
            self.db.trim_embeddings(emp_id, cap)
        return True


# ---------- voice ----------------------------------------------------------


class Greeter:
    """Async-TTS-backed greeter with per-emp_id cooldown."""

    def __init__(self, async_tts: AsyncTTS):
        self._tts = async_tts
        self._last_greeted: dict[str, float] = {}

    def greet(self, emp_id: str, name: str) -> bool:
        now = time.time()
        if now - self._last_greeted.get(emp_id, 0.0) < config.GREET_COOLDOWN_SEC:
            return False
        self._last_greeted[emp_id] = now
        msg = f"Hello {name}, welcome!"
        print(f"[GREET] {msg}")
        self._tts.speak(msg)
        return True

    def say(self, text: str) -> None:
        print(f"[TTS] {text}")
        self._tts.speak(text)

    def reset_last(self) -> None:
        self._last_greeted.clear()


# ---------- camera ---------------------------------------------------------


def open_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


def grab_frame(cam: Picamera2) -> np.ndarray:
    return cam.capture_array()


def resize_to_camera_pane(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox the camera frame into a (width x height) pane."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_off = (width - new_w) // 2
    y_off = (height - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ---------- HUD draw on camera frame ---------------------------------------


def draw_face_overlays(frame: np.ndarray, dets, label_for) -> None:
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = label_for(d)
        if label:
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ---------- right-panel state machine --------------------------------------


class RightPanel:
    """Tabbed right-side panel: PROJECTS / REGISTER / CHAT."""

    def __init__(self, db: FaceDB, chat: ChatClient):
        self.db = db
        self.chat = chat
        self.tab = "PROJECTS"
        self.projects: list = []
        self.projects_loaded_at = 0.0
        self.selected_idx = 0
        self.scroll = 0
        self.register_form = views.RegisterFormState()
        self.register_message = ""
        self.chat_history: list[tuple[str, str]] = []
        self.chat_input_mode = config.CHAT_VOICE_MODE_DEFAULT
        self.chat_partial = ""
        self.chat_pending_future: Future | None = None
        self.chat_pending_started_at = 0.0
        self.chat_listening = False

    # ---- session lifecycle --------------------------------------------

    def session_reset(self) -> None:
        self.tab = "PROJECTS"
        self.selected_idx = 0
        self.scroll = 0
        self.register_form = views.RegisterFormState()
        self.register_message = ""
        self.chat_history = []
        self.chat_input_mode = config.CHAT_VOICE_MODE_DEFAULT
        self.chat_partial = ""
        self.chat_pending_future = None
        self.chat_listening = False

    def reload_projects(self) -> None:
        self.projects = self.db.list_projects()
        self.projects_loaded_at = time.time()

    # ---- tab switching ------------------------------------------------

    def set_tab(self, tab: str) -> None:
        if tab not in ("PROJECTS", "REGISTER", "CHAT"):
            return
        if self.tab == "CHAT" and tab != "CHAT":
            # Leaving chat -- the main loop will read self.chat_listening
            # and call stop_chat_listening() to revert the recognizer.
            self.chat_listening = False
        self.tab = tab
        if tab == "PROJECTS":
            self.reload_projects()

    # ---- rendering ----------------------------------------------------

    def render(self, session_count: int) -> np.ndarray:
        size = (config.PANEL_WIDTH, config.WINDOW_SIZE[1])
        # Refresh project list every ~5 s so admin edits show up.
        if (self.tab == "PROJECTS"
                and time.time() - self.projects_loaded_at > 5):
            self.reload_projects()

        if self.tab == "PROJECTS":
            return views.render_projects_panel(
                self.projects, self.selected_idx, self.scroll,
                session_count, size,
            )
        if self.tab == "REGISTER":
            return views.render_register_panel(
                self.register_form, self.register_message,
                session_count, size,
            )
        if self.tab == "CHAT":
            remaining = self.chat.budget.remaining(self._chat_emp_id)
            partial = self.chat_partial
            if self.chat_pending_future is not None:
                dots = "." * (int(time.time() * 2) % 4)
                partial = f"thinking{dots}"
            elif self.chat_listening:
                dots = "." * (int(time.time() * 2) % 4)
                partial = f"listening{dots} (tap or press space to stop)"
            return views.render_chat_panel(
                self.chat_history, remaining, self.chat_input_mode,
                partial, self.chat.status(),
                session_count, size,
            )
        return views.render_projects_panel([], 0, 0, session_count, size)

    # ---- mouse / touch input ------------------------------------------

    def handle_click(self, x: int, y: int) -> dict:
        """Click given in panel-local coordinates (origin top-left of the
        right pane). Returns the same shape of dict as handle_key."""
        out: dict = {}
        # Tab strip is the top ~38 px of the panel.
        if y < 38:
            tabs = ["PROJECTS", "REGISTER", "CHAT"]
            tab_w = config.PANEL_WIDTH // len(tabs)
            idx = max(0, min(len(tabs) - 1, x // tab_w))
            self.set_tab(tabs[idx])
            out["activity"] = True
            return out
        # CHAT tab: clicks in the bottom input row.
        # Mic / keyboard toggle button on the right ~60px.
        # The rest of the input box toggles voice listening (in voice mode).
        if self.tab == "CHAT" and y > config.WINDOW_SIZE[1] - 70:
            out["activity"] = True
            btn_left = config.PANEL_WIDTH - 12 - 60
            if x >= btn_left:
                # Toggle input mode and stop any in-flight listening.
                self.chat_input_mode = (
                    "keyboard" if self.chat_input_mode == "voice" else "voice"
                )
                self.chat_partial = ""
                if self.chat_listening:
                    out["stop_listening"] = True
                return out
            if self.chat_input_mode == "voice":
                if self.chat_listening:
                    out["stop_listening"] = True
                else:
                    out["start_listening"] = True
        return out

    # ---- keyboard input ----------------------------------------------

    _chat_emp_id: str = "anon"

    def set_chat_emp_id(self, emp_id: str) -> None:
        self._chat_emp_id = emp_id

    def handle_key(self, key: int) -> dict:
        """Returns a dict describing side effects, e.g.
        {"submit_register": True} or {"submit_chat": "..."} so the main
        loop can run blocking flows (capture poses, show TTS replies)."""
        out: dict = {}
        if key == -1:
            return out
        # Tab switching with single keys (only when not focused on a text field)
        if self.tab == "REGISTER":
            return self._handle_key_register(key)
        if self.tab == "CHAT":
            return self._handle_key_chat(key)
        # PROJECTS tab navigation.
        if key == ord("p") or key == ord("P"):
            self.set_tab("PROJECTS")
        elif key == ord("r") or key == ord("R"):
            self.set_tab("REGISTER")
        elif key == ord("c") or key == ord("C"):
            self.set_tab("CHAT")
        elif key == 82:   # up arrow (linux)
            self.selected_idx = max(0, self.selected_idx - 1)
            if self.selected_idx < self.scroll:
                self.scroll = self.selected_idx
        elif key == 84:   # down arrow
            self.selected_idx = min(len(self.projects) - 1,
                                    self.selected_idx + 1)
        return out

    def _handle_key_register(self, key: int) -> dict:
        out: dict = {}
        f = self.register_form
        if key == 27:           # Esc -> back to projects
            self.set_tab("PROJECTS")
            self.register_form = views.RegisterFormState()
            self.register_message = ""
            return out
        if key == 9:            # Tab -> switch field
            f.focused = "name" if f.focused == "emp_id" else "emp_id"
            return out
        if key in (10, 13):     # Enter -> submit
            if not f.emp_id.strip():
                self.register_message = "Please enter your Employee ID first."
                return out
            if not f.name.strip():
                self.register_message = "Please enter your name to continue."
                return out
            out["submit_register"] = True
            return out
        if key == 8 or key == 127:  # Backspace
            cur = getattr(f, f.focused)
            setattr(f, f.focused, cur[:-1])
            return out
        if 32 <= key < 127:
            cur = getattr(f, f.focused)
            setattr(f, f.focused, cur + chr(key))
        return out

    def _handle_key_chat(self, key: int) -> dict:
        out: dict = {}
        if key == 27:           # Esc -> back to projects
            self.set_tab("PROJECTS")
            self.chat_partial = ""
            out["stop_listening"] = True
            return out
        if key == ord("v") or key == ord("V"):
            self.chat_input_mode = (
                "keyboard" if self.chat_input_mode == "voice" else "voice"
            )
            self.chat_partial = ""
            out["stop_listening"] = True
            return out
        if self.chat_input_mode == "voice":
            # Space toggles voice listening on/off so the user can talk.
            if key == ord(" "):
                if self.chat_listening:
                    out["stop_listening"] = True
                else:
                    out["start_listening"] = True
            return out
        # Keyboard mode below.
        if key == 8 or key == 127:
            self.chat_partial = self.chat_partial[:-1]
            return out
        if key in (10, 13):
            q = self.chat_partial.strip()
            if q:
                out["submit_chat"] = q
                self.chat_partial = ""
            return out
        if 32 <= key < 127:
            self.chat_partial += chr(key)
        return out


# ---------- voice-guided pose capture -------------------------------------


def capture_with_prompts(
    cam: Picamera2, pipe: HailoFacePipeline, greeter: Greeter,
    tts: AsyncTTS, render_frame=None,
) -> list[np.ndarray]:
    """If render_frame is given, it's called every iteration with
    (frame, det_or_None, prompt_str, pose_idx, total_poses, status_str).
    The callback is responsible for drawing to the cv2 window and
    handling waitKey -- that keeps the kiosk display responsive while
    pose capture runs."""
    embeddings: list[np.ndarray] = []
    prompts = config.POSE_PROMPTS
    baseline_anchor = None
    baseline_eye_dist = 1.0

    for i, (prompt, direction) in enumerate(prompts, start=1):
        greeter.say(prompt)
        # Wait for the prompt to finish playing before we start watching
        # for the user's pose change. Render every frame while waiting so
        # the camera preview keeps updating during the audio.
        wait_deadline = time.time() + 8
        while not tts.wait_idle(timeout=0.05) and time.time() < wait_deadline:
            if render_frame is not None:
                render_frame(grab_frame(cam), None, prompt, i, len(prompts),
                             "speaking...")
        hold_until = time.time() + config.POSE_HOLD_SEC
        deadline = time.time() + config.POSE_HOLD_SEC + config.POSE_CAPTURE_TIMEOUT_SEC
        stable_since = None
        last_lms = None
        captured = False

        while not captured and time.time() < deadline:
            frame = grab_frame(cam)
            dets = pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                               config.DETECTOR_NMS_IOU)
            det = largest_detection(dets)
            ok = False
            if det is not None:
                ok, _ = is_quality_face(det, frame.shape)
            status = "looking for face..."
            if det is None:
                status = "step into frame"
            elif not ok:
                status = "move closer / face the camera"
            elif time.time() < hold_until:
                status = "hold steady..."
            else:
                status = "capturing — hold the pose"
            if render_frame is not None:
                render_frame(frame, det, prompt, i, len(prompts), status)

            if ok and time.time() >= hold_until:
                anchor, eye_dist = landmark_anchor(det)
                shift_ok = (
                    direction is None
                    or baseline_anchor is None
                    or shift_matches_direction(
                        anchor, baseline_anchor, baseline_eye_dist, direction
                    )
                )
                if shift_ok:
                    if last_lms is not None and landmarks_drift(det.landmarks, last_lms) <= config.POSE_STABLE_PIXEL_TOL:
                        if stable_since is None:
                            stable_since = time.time()
                        elif time.time() - stable_since >= config.POSE_STABLE_SEC:
                            aligned = align_face(frame, det.landmarks)
                            embeddings.append(pipe.embed(aligned))
                            if direction is None and baseline_anchor is None:
                                baseline_anchor = anchor
                                baseline_eye_dist = eye_dist
                            captured = True
                    else:
                        stable_since = None
                    last_lms = det.landmarks
                else:
                    stable_since = None
                    last_lms = det.landmarks
            else:
                stable_since = None
                last_lms = None

        if not captured:
            greeter.say("That's okay! Let's try the next one.")
            tts.wait_idle(timeout=4)
    return embeddings


# ---------- registration flow (tab-driven) --------------------------------


def run_in_panel_registration(
    panel: RightPanel, db: FaceDB, greeter: Greeter, tts: AsyncTTS,
    cam: Picamera2, pipe: HailoFacePipeline,
    render_capture=None,
) -> bool:
    """Called when the REGISTER tab submits. Captures poses and writes
    to the DB. Returns True on successful (re-)registration.

    `render_capture` is forwarded into capture_with_prompts so the cv2
    window keeps updating during the multi-second pose loop."""
    f = panel.register_form
    f.busy = True
    panel.register_message = "Capturing your photo — follow the friendly voice prompts!"

    is_existing = db.employee_exists(f.emp_id)
    if is_existing:
        existing_name = db.get_name(f.emp_id) or ""
        greeter.say(f"Welcome back, {existing_name}! Just confirming it's you.")
    else:
        existing_name = f.name

    embeddings = capture_with_prompts(cam, pipe, greeter, tts, render_capture)
    if not embeddings:
        panel.register_message = ("Hmm, we couldn't capture a clear photo. "
                                   "Take a step closer and try again.")
        greeter.say("Let's give that another go.")
        f.busy = False
        return False

    if is_existing:
        emp_ids, _, matrix = db.load_all()
        score = best_self_match(embeddings, emp_ids, matrix, f.emp_id)
        if score < config.REREGISTER_MATCH_THRESHOLD:
            panel.register_message = ("Looks like a fresh face! "
                                       "Try a new Employee ID to register.")
            greeter.say("Let's set you up with a new profile.")
            f.busy = False
            return False
        for e in embeddings:
            db.add_embedding(f.emp_id, e)
        greeter.say(f"Wonderful, I've added {len(embeddings)} new looks for you.")
        panel.register_message = f"Welcome back, {existing_name}! All updated."
    else:
        db.add_employee(f.emp_id, f.name, embeddings)
        greeter.say(f"Welcome aboard, {f.name}! Lovely to meet you.")
        panel.register_message = f"Welcome, {f.name}! You are all set."

    f.busy = False
    panel.set_tab("PROJECTS")  # back to default after success
    return True


# ---------- main loop -----------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--auto-register", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    args = parser.parse_args()
    show_preview = not args.no_display

    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    db = FaceDB()
    weather = WeatherPoller()
    weather.start()
    chat = ChatClient()
    backend = make_backend()
    tts = AsyncTTS(backend)
    greeter = Greeter(tts)
    cam = open_camera()
    liveness = LivenessChecker()
    learner = SilentLearner(db)
    blinker = BlinkChecker(pipe, grab_frame, lambda *a, **kw: None)
    panel = RightPanel(db, chat)
    panel.reload_projects()

    listener: WakeWordListener | None = None
    state = "ACTIVE" if args.no_wake_word else "IDLE"
    if not args.no_wake_word:
        try:
            listener = WakeWordListener(
                config.VOSK_MODEL_DIR, config.WAKE_WORD,
                samplerate=config.WAKE_WORD_SAMPLERATE,
                blocksize=config.WAKE_WORD_BLOCKSIZE,
            )
            listener.start()
            print(f"[wake-word] listening for: '{config.WAKE_WORD}'")
        except WakeWordError as exc:
            print(f"[wake-word] disabled: {exc}")
            state = "ACTIVE"

    emp_ids, names, matrix = db.load_all()
    print(f"Loaded {matrix.shape[0]} embeddings for {len(set(emp_ids))} employees.")
    print(f"Initial state: {state}")

    unknown_streak = 0
    last_interaction_at = 0.0
    activated_at = 0.0
    blink_confirmed: set[str] = set()
    session_id = ""
    seen_in_session: set[str] = set()
    last_known_emp_id = "anon"

    # ---- mouse / touch ---------------------------------------------------
    # Single-element list so the cv2 callback (running on the GUI thread)
    # can hand a click off to the main loop without locking. Touchscreens
    # deliver button events as mouse events, so this covers both inputs.
    pending_click: list = [None]

    def on_mouse(event, mx, my, flags, _param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click[0] = (mx, my)

    if show_preview:
        cv2.namedWindow("Echo AI", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Echo AI", on_mouse)

    # ---- chat voice (free-form ASR) -------------------------------------
    # Final transcripts from the wake-word listener (when in freeform mode)
    # land here; the main loop drains the queue and submits them as chat
    # questions.
    chat_voice_q: queue.Queue = queue.Queue()

    def _on_chat_voice_final(text: str) -> None:
        chat_voice_q.put(text)

    def start_chat_listening() -> None:
        if listener is None:
            return
        if panel.chat_listening:
            return
        listener.set_freeform(_on_chat_voice_final)
        panel.chat_listening = True

    def stop_chat_listening() -> None:
        if listener is None:
            panel.chat_listening = False
            return
        if not panel.chat_listening:
            return
        listener.set_wake()
        panel.chat_listening = False

    def go_active(reason: str) -> None:
        nonlocal state, activated_at, last_interaction_at, session_id
        state = "ACTIVE"
        activated_at = time.time()
        last_interaction_at = activated_at
        if listener is not None:
            listener.deactivate()
        greeter.reset_last()
        liveness.reset()
        blink_confirmed.clear()
        seen_in_session.clear()
        session_id = uuid.uuid4().hex[:12]
        chat.budget.reset()
        panel.session_reset()
        panel.reload_projects()
        tts.flush()
        greeter.say("Hello! Lovely to see you.")
        print(f"[state] IDLE -> ACTIVE via {reason} (session {session_id})")

    try:
        while True:
            frame = grab_frame(cam)

            # ---- mouse / touch input -------------------------------
            click = pending_click[0]
            if click is not None:
                pending_click[0] = None
                if state == "IDLE":
                    # Touch / click anywhere on the welcome screen wakes
                    # the kiosk (same effect as the wake word). Useful
                    # when Vosk is offline or the visitor is in a
                    # noisy environment.
                    go_active("touch")
                elif show_preview:
                    # Translate window pixels to logical (un-scaled)
                    # coords, then split into camera vs panel.
                    scale = config.DISPLAY_SCALE or 1.0
                    wx = int(click[0] / scale)
                    wy = int(click[1] / scale)
                    cam_w = config.WINDOW_SIZE[0] - config.PANEL_WIDTH
                    if wx >= cam_w:
                        actions = panel.handle_click(wx - cam_w, wy)
                        if actions:
                            last_interaction_at = time.time()
                        if actions.get("start_listening"):
                            start_chat_listening()
                        if actions.get("stop_listening"):
                            stop_chat_listening()

            # ---- keep listener mode in sync with panel state ------
            # If anything (tab change, idle-out, etc.) flipped
            # panel.chat_listening to False, revert the recognizer to
            # wake-word grammar. Otherwise wake-word detection won't fire.
            if listener is not None and listener.mode == "freeform" \
                    and not panel.chat_listening:
                listener.set_wake()

            # ---- drain free-form ASR results -----------------------
            try:
                while True:
                    spoken = chat_voice_q.get_nowait()
                    if state != "ACTIVE" or not spoken:
                        continue
                    stop_chat_listening()
                    panel.chat_partial = ""
                    _submit_chat(panel, chat, last_known_emp_id, spoken)
                    last_interaction_at = time.time()
            except queue.Empty:
                pass

            # ---- IDLE -> ACTIVE on wake word -------------------------
            if state == "IDLE" and listener is not None and listener.is_activated():
                go_active("wake-word")

            # ---- recognition (only in ACTIVE) -----------------------
            biggest_quality_unknown = False
            labels: dict[int, str] = {}
            dets = pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                               config.DETECTOR_NMS_IOU)
            biggest = largest_detection(dets)

            if state == "ACTIVE":
                biggest_is_live = liveness.update(frame, biggest)
                pending_greets: list[tuple[str, str, np.ndarray, float]] = []

                for i, det in enumerate(dets):
                    ok, reason = is_quality_face(det, frame.shape)
                    if not ok:
                        labels[i] = f"low quality: {reason}"
                        continue
                    if det is biggest:
                        if not biggest_is_live:
                            labels[i] = f"checking liveness..."
                            continue
                    else:
                        sf_ok, sf_reason = LivenessChecker.single_frame_check(frame, det)
                        if not sf_ok:
                            labels[i] = f"liveness: {sf_reason}"
                            continue

                    aligned = align_face(frame, det.landmarks)
                    emb = pipe.embed(aligned)
                    idx, score = cosine_match(emb, matrix)
                    if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                        labels[i] = f"{names[idx]} ({score:.2f})"
                        if learner.maybe_add(emp_ids[idx], names[idx], emb, score):
                            emp_ids, names, matrix = db.load_all()
                        pending_greets.append(
                            (emp_ids[idx], names[idx], emb, score)
                        )
                    else:
                        labels[i] = f"unknown ({score:.2f})"
                        if det is biggest:
                            biggest_quality_unknown = True

                # Blink challenge on any unconfirmed emp_id we want to greet.
                need_blink = [
                    g for g in pending_greets if g[0] not in blink_confirmed
                ]
                if config.LIVENESS_REQUIRE_BLINK and need_blink:
                    if blinker.run(cam, greeter, show_preview=False):
                        for eid, _, _, _ in need_blink:
                            blink_confirmed.add(eid)
                        last_interaction_at = time.time()
                    else:
                        greeter.say("One more blink, please!")
                        pending_greets = [
                            g for g in pending_greets if g[0] in blink_confirmed
                        ]

                for emp_id, name, _, _ in pending_greets:
                    if greeter.greet(emp_id, name):
                        last_interaction_at = time.time()
                        # Counter increments at most once per (emp_id, session)
                        if emp_id not in seen_in_session:
                            seen_in_session.add(emp_id)
                            db.record_interaction(emp_id, session_id)
                        last_known_emp_id = emp_id
                        panel.set_chat_emp_id(emp_id)

                # Auto-jump to REGISTER tab when an unknown sticks around,
                # but only if the user is on the default PROJECTS view.
                # Hijacking someone mid-chat or mid-registration is jarring
                # -- if they want to register a different person they can
                # press R themselves.
                if biggest_quality_unknown:
                    last_interaction_at = time.time()
                    unknown_streak += 1
                    if (args.auto_register
                            and unknown_streak >= config.UNKNOWN_FRAMES_BEFORE_REGISTER
                            and panel.tab == "PROJECTS"):
                        unknown_streak = 0
                        panel.set_tab("REGISTER")
                        panel.register_message = (
                            "Hi there! Pop your name and Employee ID below "
                            "and press Enter to introduce yourself."
                        )
                        greeter.say(
                            "Hello! Looks like you're new — please pop your "
                            "details into the screen so I can greet you next time."
                        )
                else:
                    unknown_streak = 0

                # Idle timeout.
                # Don't sleep while the user is on REGISTER or CHAT, or
                # while a chat reply is in flight -- those are explicit
                # signs the user is engaged even if nothing else is
                # changing on the camera side.
                user_engaged = (
                    panel.tab in ("REGISTER", "CHAT")
                    or panel.chat_pending_future is not None
                )
                if user_engaged:
                    last_interaction_at = time.time()
                idle_for = (time.time() - last_interaction_at) if last_interaction_at else 0
                if (idle_for >= config.IDLE_AFTER_LAST_INTERACTION_SEC
                        or (activated_at and time.time() - activated_at >= config.ACTIVE_SESSION_MAX_SEC)):
                    if listener is not None:
                        state = "IDLE"
                        liveness.reset()
                        greeter.reset_last()
                        listener.deactivate()
                        print(f"[state] ACTIVE -> IDLE (idle {idle_for:.0f}s, "
                              f"interactions={len(seen_in_session)})")

                # Chat result polling.
                if panel.chat_pending_future is not None and panel.chat_pending_future.done():
                    fut = panel.chat_pending_future
                    panel.chat_pending_future = None
                    try:
                        answer = fut.result()
                        panel.chat_history.append(("assistant", answer))
                        greeter.say(answer)
                        last_interaction_at = time.time()
                    except ChatBudgetError as exc:
                        panel.chat_history.append(("assistant", str(exc)))
                    except ChatBackendError as exc:
                        panel.chat_history.append((
                            "assistant",
                            "I'm having a little trouble reaching the chat "
                            f"service right now — let's try again in a moment. ({exc})"
                        ))
                    except Exception as exc:  # noqa: BLE001
                        panel.chat_history.append((
                            "assistant",
                            f"Hmm, something went sideways: {exc}. "
                            "Please try again."
                        ))

            # ---- render -----------------------------------------------
            if show_preview:
                if state == "IDLE":
                    composed = views.render_idle(
                        weather.get(),
                        db.interaction_count_total(),
                        config.WINDOW_SIZE,
                    )
                else:
                    cam_w = config.WINDOW_SIZE[0] - config.PANEL_WIDTH
                    cam_h = config.WINDOW_SIZE[1]
                    cam_pane = resize_to_camera_pane(frame, cam_w, cam_h)
                    # Rescale detection coords from source frame to pane.
                    pane_dets = _rescale_dets(dets, frame.shape, cam_pane.shape)
                    draw_face_overlays(
                        cam_pane, pane_dets,
                        lambda d: labels.get(pane_dets.index(d), ""),
                    )
                    panel_img = panel.render(len(seen_in_session))
                    composed = views.render_active(cam_pane, panel_img)

                if config.DISPLAY_SCALE != 1.0:
                    new_w = max(1, int(composed.shape[1] * config.DISPLAY_SCALE))
                    new_h = max(1, int(composed.shape[0] * config.DISPLAY_SCALE))
                    composed = cv2.resize(composed, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)
                cv2.imshow("Echo AI", composed)
                raw = cv2.waitKey(1)
                key = raw & 0xFF if raw != -1 else 0xFF
                if key == ord("q"):
                    break
                if state == "ACTIVE" and key != 0xFF:
                    # Any keystroke counts as user activity -- typing in
                    # CHAT or REGISTER must keep the kiosk awake even if
                    # nobody is in front of the camera.
                    last_interaction_at = time.time()
                    actions = panel.handle_key(key)
                    if actions.get("start_listening"):
                        start_chat_listening()
                    if actions.get("stop_listening"):
                        stop_chat_listening()
                    if actions.get("submit_register"):
                        def _render_capture(f, det, prompt, idx, total, status):
                            panel.register_message = (
                                f"Pose {idx}/{total}: {prompt}\n{status}"
                            )
                            cam_w = config.WINDOW_SIZE[0] - config.PANEL_WIDTH
                            cam_h = config.WINDOW_SIZE[1]
                            cam_pane = resize_to_camera_pane(f, cam_w, cam_h)
                            if det is not None:
                                pane_dets = _rescale_dets(
                                    [det], f.shape, cam_pane.shape,
                                )
                                draw_face_overlays(
                                    cam_pane, pane_dets, lambda _d: status,
                                )
                            panel_img = panel.render(len(seen_in_session))
                            composed = views.render_active(cam_pane, panel_img)
                            if config.DISPLAY_SCALE != 1.0:
                                nw = max(1, int(composed.shape[1] * config.DISPLAY_SCALE))
                                nh = max(1, int(composed.shape[0] * config.DISPLAY_SCALE))
                                composed = cv2.resize(
                                    composed, (nw, nh),
                                    interpolation=cv2.INTER_AREA,
                                )
                            cv2.imshow("Echo AI", composed)
                            cv2.waitKey(1)
                        run_in_panel_registration(
                            panel, db, greeter, tts, cam, pipe, _render_capture,
                        )
                        emp_ids, names, matrix = db.load_all()
                        last_interaction_at = time.time()
                    if (q := actions.get("submit_chat")):
                        _submit_chat(panel, chat, last_known_emp_id, q)
                        last_interaction_at = time.time()
                elif state == "IDLE" and key == ord(" "):
                    # Space bar in IDLE wakes the kiosk too -- handy when
                    # there's no mic or you're testing.
                    go_active("space-key")
            else:
                # Headless: just keep the loop running.
                time.sleep(0.02)
    finally:
        try:
            stop_chat_listening()
        except Exception:
            pass
        if listener is not None:
            listener.stop()
        weather.stop()
        chat.shutdown()
        tts.stop()
        cam.stop()
        pipe.close()
        db.close()
        if show_preview:
            cv2.destroyAllWindows()


def _is_staff_meeting_query(text: str) -> bool:
    """True for "do you have a message for the staff meeting?" and similar
    phrasings (team meeting / all-hands / town hall / msg-for-the-meeting).
    Answered locally from config.STAFF_MEETING_MESSAGE, no LLM call."""
    t = (text or "").lower()
    return any(p in t for p in (
        "staff meeting", "team meeting", "all hands", "all-hands",
        "town hall",
        "msg for the staff", "msg for the team", "msg for the meeting",
        "message for the staff", "message for the team",
        "message for the meeting", "staff msg", "team msg",
        "message for staff", "message for team",
    ))


def _submit_chat(panel: RightPanel, chat: ChatClient, emp_id: str,
                 question: str) -> None:
    panel.chat_history.append(("user", question))
    # Static-reply intercept: "do you have a msg for the staff meeting?"
    # gets the canned message from config (no LLM, no budget slot). Routed
    # through a pre-resolved Future so the existing polling path picks it
    # up, appends it to the transcript, and speaks it like any other reply.
    if _is_staff_meeting_query(question):
        msg = (getattr(config, "STAFF_MEETING_MESSAGE", "") or "").strip()
        if msg:
            f: Future = Future()
            f.set_result(msg)
            panel.chat_pending_future = f
            panel.chat_pending_started_at = time.time()
            return
    if chat.budget.remaining(emp_id) <= 0:
        panel.chat_history.append((
            "assistant",
            f"Lovely chatting! That's {config.CHAT_MAX_QUESTIONS_PER_SESSION} "
            "questions for now — come say hi again anytime."
        ))
        return
    panel.chat_pending_future = chat.submit(emp_id, question)
    panel.chat_pending_started_at = time.time()


def _rescale_dets(dets, src_shape, dst_shape):
    """Scale detection bboxes / landmarks from the source frame coords to
    the resized camera pane so overlays line up."""
    if not dets:
        return dets
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    x_off = (dst_w - new_w) / 2
    y_off = (dst_h - new_h) / 2

    out = []
    for d in dets:
        # Mutate-shallow-copy via a SimpleNamespace-like wrapper isn't
        # worth the complexity — reuse the dataclass.
        from copy import copy
        d2 = copy(d)
        x1, y1, x2, y2 = d.bbox
        d2.bbox = (
            x1 * scale + x_off, y1 * scale + y_off,
            x2 * scale + x_off, y2 * scale + y_off,
        )
        d2.landmarks = d.landmarks * scale + np.array([x_off, y_off])
        out.append(d2)
    return out


if __name__ == "__main__":
    main()
