"""ECHO SCOPE kiosk — orchestrator.

Spawns the camera worker, the weather poller, the fun-fact rotator,
the wake-word listener, the chat-voice capture, and the Flask app
that serves the browser SPA. Each component talks through the
StateBus, which Flask exposes to the browser over Server-Sent Events.

There is no more cv2 window -- the UI is a Chromium kiosk pointed at
http://127.0.0.1:8080. start.sh launches the browser after this
process is up.
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import threading
import time
import uuid

import numpy as np
from picamera2 import Picamera2

import config
import messages
from asr import make_chat_asr
from async_tts import AsyncTTS
from chat import ChatBackendError, ChatBudgetError, ChatClient
from chat_voice import ChatVoiceCapture, ChatVoiceCaptureError
from database import FaceDB
from frame_streamer import FrameStreamer
from fun_facts import FunFactRotator
from hailo_infer import HailoFacePipeline, align_face
from liveness import LivenessChecker
from quality import (is_quality_face, landmark_anchor, landmarks_drift,
                     shift_matches_direction)
import solutions as solutions_match
from state import StateBus
from tts import make_backend
from wake_word import WakeWordError, WakeWordListener
from weather import WeatherPoller
from web import create_app


# ---------- helpers --------------------------------------------------------


def cosine_match(query: np.ndarray, matrix: np.ndarray):
    """Return (best_idx, best_score, second_best_score). When fewer
    than two embeddings exist, second_best is -1.0 so margin checks
    against it always pass."""
    if matrix.shape[0] == 0:
        return -1, 0.0, -1.0
    sims = matrix @ query
    order = np.argsort(sims)[::-1]
    best_idx = int(order[0])
    best = float(sims[best_idx])
    second = float(sims[order[1]]) if order.size > 1 else -1.0
    return best_idx, best, second


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


def _draw_overlays(frame, face_labels):
    """Annotate `frame` in-place with rectangles + name labels.

    face_labels: list of (det, label_text, bgr_color) tuples produced
    by the recognition loop. Drawing happens server-side -- the
    browser just decodes the resulting JPEG -- so the user gets the
    "boxes around faces" visual that the cv2 build had, without
    spending bandwidth on per-detection JSON pushes."""
    import cv2
    for det, text, color in face_labels:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not text:
            continue
        (tw, th), _bl = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1,
        )
        # Label background bar just above the box.
        ly2 = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, ly2), (x1 + tw + 12, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )


def _is_goodbye(text: str) -> bool:
    """Return True if the transcribed user utterance signals end-of-chat.

    Normalisation: lowercase, strip leading/trailing whitespace, replace
    every punctuation character with a space, collapse runs of
    whitespace. So "Bye, thank you!" -> "bye thank you" -- matches the
    token "bye thank you" directly.

    Then matches when the normalised utterance is EITHER:
      - exactly a goodbye token, OR
      - ends with " <token>"   ("alright bye"   -> matches "bye"), OR
      - starts with "<token> " AND the utterance is <= 5 words long
        ("bye thank you" matches, but "bye, I have one more question
        to ask" does NOT -- 9 words, falls past the short-utterance
        cap).

    The short-utterance cap is the cheapest defence against a user
    starting a long question with a polite "Bye" and then continuing.
    """
    import re
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s']+", " ", t)        # punctuation -> space (keep apostrophe)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    word_count = len(t.split())
    tokens = getattr(config, "CHAT_GOODBYE_TOKENS", ())
    for tok in tokens:
        tok = tok.lower().strip()
        if not tok:
            continue
        # Exact match + endswith always fire -- the goodbye word is at
        # the end of the sentence, which is the strongest signal.
        if t == tok or t.endswith(" " + tok):
            return True
        # Startswith only fires on short utterances to avoid cutting
        # off a user who began with a polite "bye" and is about to ask
        # something else.
        if word_count <= 5 and t.startswith(tok + " "):
            return True
    return False


def _is_weather_query(text: str) -> bool:
    """Return True for questions about the *current local* weather.

    Deliberately conservative so it never hijacks broader questions
    ("what is climate change?", "weather patterns in the 1800s"). We
    answer these from the kiosk's own weather poller (state.weather)
    instead of the LLM -- it's instant, free, and always correct for
    the kiosk's location, and keeps the answer to one crisp line.
    """
    t = (text or "").lower()
    if "weather" in t or "temperature" in t or "forecast" in t:
        return True
    if any(p in t for p in ("how hot", "how cold", "how warm")):
        return True
    if any(p in t for p in (
        "is it raining", "is it snowing", "is it sunny",
        "raining outside", "snowing outside",
    )):
        return True
    return False


def _is_project_list_query(text: str) -> bool:
    """Return True for "what projects / AI files / demos are on display
    today?"-style listing questions. We answer these straight from the
    project list shown on the dashboard (db/state.projects) so the kiosk
    reads out exactly what's on display -- never a hallucinated extra.
    Detail questions ("tell me about project X") fall through to the LLM,
    which still gets the project list folded into its system prompt."""
    t = (text or "").lower()
    return any(p in t for p in (
        "what project", "which project", "what projects", "what are the project",
        "projects available", "projects on display", "project on display",
        "what ai file", "which ai file", "what ai files", "ai files on display",
        "what demo", "which demo", "what demos",
        "what's on display", "whats on display", "what is on display",
        "what can i see", "what can we see",
        "list the project", "show me the project",
    ))


def _is_solution_list_query(text: str) -> bool:
    """General "what tools / solutions were developed?" -- answered by
    listing the catalog names rather than a single match."""
    t = (text or "").lower()
    return any(p in t for p in (
        # solutions ...
        "what solutions", "which solutions", "list of solutions",
        "list the solutions", "all solutions", "all the solutions",
        "solutions you have", "solutions you've", "solutions have you",
        "solutions developed", "solutions you developed", "solutions were",
        # tools ...
        "what tools", "which tools", "list of tools", "list the tools",
        "all tools", "all the tools", "tools developed", "tools you developed",
        "tools were", "tools you have", "tools have you",
        # apps / platforms / products ...
        "what apps", "which apps", "what platforms", "which platforms",
        "what products", "which products",
        # generic "what was/were built/developed/created" ...
        "what have you built", "what have you developed",
        "what did you build", "what did you develop",
        "what was built", "what was developed", "what were built",
        "what were developed", "what have you created", "what was created",
    ))


def _is_solution_query(text: str) -> bool:
    """Is the visitor asking whether we have a solution/tool for some need?
    Broad on purpose -- when nothing matches the catalog we fall back to
    the LLM, so a false positive is harmless."""
    t = (text or "").lower()
    return any(p in t for p in (
        "solution", "do you have", "do we have", "is there a tool",
        "is there a platform", "is there an app", "any tool", "any platform",
        "anything for", "tool for", "platform for", "app for", "product for",
        "have you built", "have you developed", "did you build",
        "did you develop", "looking for a", "something for",
    ))


# ---------- silent learning -----------------------------------------------


class SilentLearner:
    def __init__(self, db: FaceDB):
        self.db = db
        self._last_added: dict[str, float] = {}

    def maybe_add(self, emp_id, name, embedding, score,
                  runner_up: float = -1.0) -> bool:
        if not config.SILENT_LEARN_ENABLED:
            return False
        if score < config.SILENT_LEARN_MIN_SCORE:
            return False
        # The classic drift trap: two similar-looking faces both score
        # 0.65-ish against the same person. Both look "comfortable" in
        # isolation, but the small margin says the model isn't actually
        # sure which one this is. Refuse to learn when the runner-up is
        # too close.
        if runner_up >= 0.0 and (score - runner_up) < config.SILENT_LEARN_MIN_MARGIN:
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
        if self.db.count_embeddings(emp_id) > cap:
            self.db.trim_embeddings(emp_id, cap)
        return True


# ---------- voice ---------------------------------------------------------


class Greeter:
    """Async TTS-backed greeter with per-emp_id cooldown + a toast push
    to the SPA so the user can see who was greeted.

    Welcome text is chosen by preference:
      1. employees.custom_welcome  -- admin-written one-liner
      2. employees.welcome_cache   -- LLM-generated from profile_url
      3. random recognized_greeting from messages.py
    """

    def __init__(self, tts: AsyncTTS, state: StateBus, db=None):
        self._tts = tts
        self._state = state
        self._db = db
        self._last_greeted: dict[str, float] = {}

    def _pick_welcome(self, emp_id: str, name: str) -> str:
        if self._db is not None:
            profile = self._db.get_employee_profile(emp_id)
            if profile:
                custom = (profile.get("custom_welcome") or "").strip()
                if custom:
                    return custom.format(name=name) if "{name}" in custom else custom
                cached = (profile.get("welcome_cache") or "").strip()
                if cached:
                    return cached
        return messages.random_recognized_greeting(name)

    def greet(self, emp_id: str, name: str) -> bool:
        now = time.time()
        if now - self._last_greeted.get(emp_id, 0.0) < config.GREET_COOLDOWN_SEC:
            return False
        self._last_greeted[emp_id] = now
        msg = self._pick_welcome(emp_id, name)
        print(f"[GREET] {msg}", flush=True)
        # Push the toast + person state from the TTS worker thread so
        # the UI update lands exactly when audio starts playing, not
        # N queued utterances earlier.
        t_queued = time.time()
        def on_start():
            wait_ms = (time.time() - t_queued) * 1000
            print(f"[tts] greet on-air after {wait_ms:.0f}ms in queue",
                  flush=True)
            self._state.update(
                person={"emp_id": emp_id, "name": name},
                toast={"text": msg, "since": time.time()},
            )
        self._tts.speak(msg, on_start=on_start)
        return True

    def say(self, text: str) -> None:
        snip = (text[:60] + "...") if len(text) > 63 else text
        print(f"[tts] queued: {snip!r}", flush=True)
        t_queued = time.time()
        def on_start():
            wait_ms = (time.time() - t_queued) * 1000
            print(f"[tts] on-air after {wait_ms:.0f}ms in queue", flush=True)
            self._state.update(toast={"text": text, "since": time.time()})
        self._tts.speak(text, on_start=on_start)

    def reset_last(self) -> None:
        self._last_greeted.clear()


# ---------- camera --------------------------------------------------------


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


# ---------- camera worker thread ------------------------------------------


class CameraWorker(threading.Thread):
    """Owns the camera, runs detection / recognition / registration, and
    publishes the latest frame to FrameStreamer and state changes to
    the StateBus."""

    daemon = True

    def __init__(self, db, state, frames, tts, chat, greeter, learner,
                 register_queue, chat_queue, wake_event,
                 idle_event, args, listener=None):
        super().__init__(name="camera-worker")
        self.db = db
        self.state = state
        self.frames = frames
        self.tts = tts
        self.chat = chat
        self.greeter = greeter
        self.learner = learner
        self.register_q = register_queue
        self.chat_q = chat_queue
        self.wake_event = wake_event
        self.idle_event = idle_event
        self.listener = listener
        self.args = args
        self.cam = None
        self.pipe = None
        self.liveness = LivenessChecker()
        self._stop = threading.Event()
        # Heartbeat: stamped each loop iteration (and during long
        # blocking ops like registration). The watchdog restarts the
        # process if this goes stale -- see _start_watchdog in main().
        self.last_loop_at = time.time()
        # When the user explicitly says "no thanks" to registration, we
        # don't re-pop the overlay for this long. Otherwise the unknown-
        # face streak would re-open it immediately.
        self._register_declined_at = 0.0
        # The first person greeted in an ACTIVE session "owns" it. We
        # won't greet anyone else until the session ends, so a colleague
        # who wanders into frame mid-conversation doesn't get called out
        # over the current user. Reset to None on every IDLE -> ACTIVE.
        self._session_primary: str | None = None

    def stop(self):
        self._stop.set()

    # ---- hardware lifecycle ----

    def _cleanup(self):
        """Release the camera + Hailo pipeline. Safe to call repeatedly."""
        if self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
        if self.pipe is not None:
            try:
                self.pipe.close()
            except Exception:
                pass
            self.pipe = None

    def _open_hardware(self):
        """(Re)open the Hailo pipeline + camera, releasing any prior
        handles first so a reopen can't leak the NPU / camera device."""
        self._cleanup()
        self.pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
        self.cam = open_camera()

    # ---- the loop ----

    def run(self):
        # Init with retry + backoff. systemd (Restart=always) is the
        # ultimate backstop, but a few retries here ride out transient
        # boot races (camera not enumerated yet, NPU busy from a prior
        # run) without a full process bounce.
        retries = getattr(config, "WORKER_INIT_RETRIES", 5)
        for attempt in range(1, retries + 1):
            try:
                self._open_hardware()
                break
            except Exception as exc:
                print(f"[camera-worker] init attempt {attempt}/{retries} "
                      f"failed: {exc!r}", flush=True)
                self._cleanup()
                if self._stop.is_set():
                    return
                time.sleep(min(2 ** attempt, 16))
        else:
            print("[camera-worker] init failed after retries; exiting for "
                  "restart", flush=True)
            os._exit(1)
        emp_ids, names, matrix = self.db.load_all()
        print(f"Loaded {matrix.shape[0]} embeddings for {len(set(emp_ids))} employees.")
        self._refresh_idle_data()

        # State machine local vars.
        kiosk_state = "ACTIVE" if self.args.no_wake_word else "IDLE"
        self.state.update(state=kiosk_state, state_since=time.time())
        unknown_streak = 0
        last_interaction_at = time.time() if kiosk_state == "ACTIVE" else 0.0
        activated_at = time.time() if kiosk_state == "ACTIVE" else 0.0
        seen_in_session: set[str] = set()
        session_id = uuid.uuid4().hex[:12] if kiosk_state == "ACTIVE" else ""
        last_known_emp_id = "anon"
        last_idle_refresh = 0.0

        while not self._stop.is_set():
            self.last_loop_at = time.time()      # watchdog heartbeat
            try:
                frame = grab_frame(self.cam)
            except Exception as exc:  # noqa: BLE001
                # Transient camera read error -- skip this frame. A
                # persistent failure stalls the heartbeat and the
                # watchdog bounces the process for a clean restart.
                print(f"[camera-worker] frame grab failed: {exc!r}", flush=True)
                time.sleep(0.1)
                continue
            self.state.set_now()
            # Detection overlays are drawn onto `frame` further down,
            # then we push to MJPEG. While IDLE we push immediately so
            # the dashboard's (unused) camera socket stays warm.
            if kiosk_state == "IDLE":
                self.frames.push(frame)

            # ---- IDLE -> ACTIVE on wake word OR touch -------
            if kiosk_state == "IDLE" and self.wake_event.is_set():
                self.wake_event.clear()
                kiosk_state = "ACTIVE"
                activated_at = time.time()
                last_interaction_at = activated_at
                seen_in_session = set()
                session_id = uuid.uuid4().hex[:12]
                self._session_primary = None
                self.greeter.reset_last()
                self.liveness.reset()
                self.chat.budget.reset()
                self.tts.flush()
                self.state.go_active()
                self.state.update(
                    person=None,
                    chat_history=[],
                    chat_remaining=config.CHAT_MAX_QUESTIONS_PER_SESSION,
                    chat_pending=False,
                )
                # Pause the wake-word listener while ACTIVE -- there's no
                # point burning CPU / contending for the mic when the
                # user is already engaged with the kiosk.
                if self.listener is not None:
                    self.listener.pause()
                self.greeter.say("Hello! Lovely to see you.")
                print(f"[state] IDLE -> ACTIVE (session {session_id})")

            # Refresh idle data periodically (projects / metrics /
            # session card) -- once every 5 s is plenty.
            if time.time() - last_idle_refresh > 5:
                last_idle_refresh = time.time()
                self._refresh_idle_data()

            # ---- process register requests (from web POST) ---
            try:
                while True:
                    req = self.register_q.get_nowait()
                    # "Skip" sentinel: user declined. Speak a decline
                    # line and close the overlay -- but stay ACTIVE so
                    # the unregistered visitor can still use chat.
                    # IDLE will fire naturally after the configured
                    # timeout if they don't engage. The 120 s
                    # register-decline cooldown prevents the form from
                    # immediately re-popping while the same unknown
                    # face is still in frame.
                    if isinstance(req, dict) and req.get("action") == "skip":
                        self._register_declined_at = time.time()
                        self.state.update(register_open=False,
                                          register_pose=None)
                        self.greeter.say(messages.random_registration_prompt(
                            "decline_registration",
                        ))
                        last_interaction_at = time.time()
                        continue
                    if kiosk_state != "ACTIVE":
                        continue
                    self._run_registration(req)
                    emp_ids, names, matrix = self.db.load_all()
                    last_interaction_at = time.time()
            except queue.Empty:
                pass

            # ---- process chat-voice transcripts --------------
            try:
                while True:
                    spoken = self.chat_q.get_nowait()
                    if kiosk_state != "ACTIVE" or not spoken:
                        continue
                    self._submit_chat(last_known_emp_id, spoken)
                    last_interaction_at = time.time()
            except queue.Empty:
                pass

            # Skip face detection / recognition while a chat exchange
            # is in flight (listening or awaiting reply). The camera
            # feed still streams to the SPA (so the user sees themselves
            # if the listen overlay isn't covering it), but no Hailo
            # cycles get spent on a face that isn't going to be acted on.
            snap_chat = self.state.snapshot()
            chat_busy = bool(
                snap_chat.get("chat_pending") or snap_chat.get("listening")
            )
            if chat_busy:
                unknown_streak = 0          # don't pop register mid-chat

            # ---- recognition (only in ACTIVE, and only while idle) ----
            biggest_quality_unknown = False
            dets = []
            biggest = None
            face_labels: list[tuple] = []
            if not chat_busy:
                dets = self.pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                                        config.DETECTOR_NMS_IOU)
                biggest = largest_detection(dets)

            if kiosk_state == "ACTIVE" and not chat_busy:
                # Liveness can be turned off entirely via config; when
                # disabled we treat every quality face as live and skip
                # both the sliding-window analysis and the per-face
                # single-frame check.
                liveness_on = getattr(config, "LIVENESS_ENABLED", True)
                biggest_is_live = (
                    self.liveness.update(frame, biggest)
                    if liveness_on else True
                )
                pending_greets = []
                for i, det in enumerate(dets):
                    ok, _ = is_quality_face(det, frame.shape)
                    if not ok:
                        face_labels.append((det, "low quality", (0, 165, 255)))
                        continue
                    if liveness_on:
                        if det is biggest:
                            if not biggest_is_live:
                                face_labels.append((det, "checking liveness…", (0, 200, 255)))
                                continue
                        else:
                            sf_ok, _ = LivenessChecker.single_frame_check(frame, det)
                            if not sf_ok:
                                face_labels.append((det, "checking liveness…", (0, 200, 255)))
                                continue
                    aligned = align_face(frame, det.landmarks)
                    emb = self.pipe.embed(aligned)
                    idx, score, runner_up = cosine_match(emb, matrix)
                    if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                        if self.learner.maybe_add(
                            emp_ids[idx], names[idx], emb, score, runner_up,
                        ):
                            emp_ids, names, matrix = self.db.load_all()
                        pending_greets.append(
                            (emp_ids[idx], names[idx], det, score)
                        )
                        face_labels.append((
                            det, f"{names[idx]}  {score:.2f}", (0, 255, 0),
                        ))
                    else:
                        face_labels.append((det, "Unknown", (0, 255, 255)))
                        if det is biggest:
                            biggest_quality_unknown = True

            # Publish to MJPEG. Overlays only when we actually ran
            # detection -- otherwise we just push the raw frame so the
            # feed doesn't go static.
            if kiosk_state == "ACTIVE":
                if not chat_busy:
                    _draw_overlays(frame, face_labels)
                self.frames.push(frame)

                for emp_id, name, _det, _score in pending_greets:
                    # Once we've greeted someone in this session, don't
                    # do it again -- even if they walk in and out of
                    # frame, the cooldown was meant to throttle spam
                    # but it still re-greets after GREET_COOLDOWN_SEC.
                    # seen_in_session is the per-session source of truth.
                    if emp_id in seen_in_session:
                        last_known_emp_id = emp_id
                        continue
                    # Greet-lock: the session's primary person has been
                    # greeted and may be mid-conversation. Don't call out
                    # anyone else who steps into frame until they leave
                    # and the session resets.
                    if (self._session_primary is not None
                            and emp_id != self._session_primary):
                        continue
                    if self.greeter.greet(emp_id, name):
                        last_interaction_at = time.time()
                        seen_in_session.add(emp_id)
                        if self._session_primary is None:
                            self._session_primary = emp_id
                        self.db.record_interaction(emp_id, session_id)
                        self._push_metrics()
                        last_known_emp_id = emp_id

                # Auto-open the register overlay for new visitors.
                snap = self.state.snapshot()
                if biggest_quality_unknown:
                    last_interaction_at = time.time()
                    unknown_streak += 1
                    declined_recently = (
                        time.time() - self._register_declined_at
                    ) < config.REGISTER_DECLINE_COOLDOWN_SEC
                    if (self.args.auto_register
                            and unknown_streak >= config.UNKNOWN_FRAMES_BEFORE_REGISTER
                            and not snap.get("register_open")
                            and not snap.get("chat_pending")
                            and not declined_recently):
                        unknown_streak = 0
                        self.state.update(
                            register_open=True,
                            register_step="form",
                            register_message=(
                                "Hi! Pop your details below and tap Register."
                            ),
                        )
                        self.greeter.say(messages.random_unrecognized_greeting())
                else:
                    unknown_streak = 0

                # Idle timeout (keep alive while user is engaged).
                # "Engaged" includes:
                #   - register overlay is open
                #   - chat is waiting for a reply
                #   - chat-voice mic is recording
                #   - a chat exchange happened recently (chat_history
                #     non-empty AND within CHAT_KEEPALIVE_SEC of last
                #     chat activity) -- gives the user time to read
                #     the answer and ask a follow-up.
                chat_active = (
                    snap.get("chat_pending")
                    or bool(snap.get("chat_history"))
                    and (time.time() - getattr(self, "_last_chat_at", 0))
                        <= config.CHAT_KEEPALIVE_SEC
                )
                user_engaged = (
                    snap.get("register_open")
                    or snap.get("listening")
                    or chat_active
                )
                if user_engaged:
                    last_interaction_at = time.time()
                idle_for = (time.time() - last_interaction_at) if last_interaction_at else 0
                force_idle = self.idle_event.is_set()
                if (force_idle
                        or idle_for >= config.IDLE_AFTER_LAST_INTERACTION_SEC
                        or (activated_at and time.time() - activated_at >= config.ACTIVE_SESSION_MAX_SEC)):
                    if force_idle:
                        self.idle_event.clear()
                    kiosk_state = "IDLE"
                    self.liveness.reset()
                    self.greeter.reset_last()
                    # Kill in-flight TTS too -- once we're on the dashboard
                    # the kiosk shouldn't keep talking about the previous
                    # session. interrupt() = backend.stop() + flush(); flush
                    # alone would only drop the queue and let the current
                    # utterance finish playing.
                    self.tts.interrupt()
                    self.state.go_idle()
                    # Resume the wake-word listener -- the user is gone,
                    # we need to be listening for the next "hello echo".
                    if self.listener is not None:
                        self.listener.resume()
                    reason = "user closed" if force_idle else f"idle {idle_for:.0f}s"
                    print(f"[state] ACTIVE -> IDLE ({reason}, "
                          f"interactions={len(seen_in_session)})")

                # Poll the chat future.
                fut = getattr(self, "_chat_future", None)
                if fut is not None and fut.done():
                    self._chat_future = None
                    # If the user already moved on -- recording a new
                    # question (listening) or tapped to listen again --
                    # the previous answer is stale. Web search can take
                    # 3-6 s, long enough that a late answer would speak
                    # over the user's next question. Discard it. Fresh
                    # snapshot so a mic-tap milliseconds ago is caught.
                    if self.state.snapshot().get("listening"):
                        print("[chat] discarding stale answer "
                               "(user is asking again)", flush=True)
                        self.state.update(chat_pending=False)
                    else:
                        try:
                            answer = fut.result()
                        except ChatBudgetError as exc:
                            answer = str(exc)
                        except ChatBackendError as exc:
                            answer = ("I'm having a little trouble reaching the "
                                      f"chat service right now ({exc}).")
                        except Exception as exc:  # noqa: BLE001
                            answer = f"Hmm, something went sideways: {exc}"
                        chat_ms = (time.time() -
                                   getattr(self, "_chat_submitted_at", time.time())
                                   ) * 1000
                        snip = (answer[:60] + "...") if len(answer) > 63 else answer
                        print(f"[chat] reply in {chat_ms:.0f}ms: {snip!r}",
                              flush=True)
                        self.state.update(chat_pending=False)
                        self._append_chat("assistant", answer)
                        self.greeter.say(answer)
                        self._last_chat_at = time.time()
                        last_interaction_at = time.time()

    # ---- registration flow (driven by /api/register) -----------------

    def _run_registration(self, req: dict):
        emp_id = req["emp_id"]
        name = req["name"]
        # Close the form overlay immediately so the user can see the
        # camera feed during pose capture. Pose progress now renders
        # as a card on the active screen via state.register_pose.
        self.state.update(
            register_open=False,
            register_step="capturing",
            register_message=f"Capturing your photo, {name}!",
        )

        is_existing = self.db.employee_exists(emp_id)
        if is_existing:
            existing_name = self.db.get_name(emp_id) or ""
            self.greeter.say(
                f"Welcome back, {existing_name}! Confirming it's you."
            )

        embeddings = self._capture_with_prompts()
        if not embeddings:
            self.state.update(
                register_step="form",
                register_message=("Hmm, we couldn't capture a clear photo. "
                                   "Take a step closer and try again."),
                register_pose=None,
            )
            self.greeter.say("Let's give that another go.")
            return

        if is_existing:
            emp_ids, _, matrix = self.db.load_all()
            score = best_self_match(embeddings, emp_ids, matrix, emp_id)
            if score < config.REREGISTER_MATCH_THRESHOLD:
                self.state.update(
                    register_step="form",
                    register_message="Looks like a fresh face! Try a new Employee ID.",
                    register_pose=None,
                )
                self.greeter.say("Let's set you up with a new profile.")
                return
            for e in embeddings:
                self.db.add_embedding(emp_id, e)
            confirm = messages.random_registration_prompt(
                "confirm_registration", name=name,
            )
            self.greeter.say(confirm)
            self.state.update(
                register_open=False,
                register_pose=None,
                register_message=f"Welcome back, {name}!",
            )
        else:
            self.db.add_employee(emp_id, name, embeddings)
            confirm = messages.random_registration_prompt(
                "confirm_registration", name=name,
            )
            self.greeter.say(confirm)
            self.state.update(
                register_open=False,
                register_pose=None,
                register_message=f"Welcome, {name}!",
            )

    def _capture_with_prompts(self):
        embeddings = []
        prompts = config.POSE_PROMPTS
        baseline_anchor = None
        baseline_eye_dist = 1.0

        for i, (prompt, direction) in enumerate(prompts, start=1):
            self.greeter.say(prompt)
            self.state.update(register_pose={
                "idx": i, "total": len(prompts),
                "prompt": prompt, "status": "speaking..."
            })
            # Render frames while the prompt plays.
            wait_deadline = time.time() + 8
            while (not self.tts.wait_idle(timeout=0.05)
                   and time.time() < wait_deadline):
                self.last_loop_at = time.time()      # watchdog heartbeat
                self.frames.push(grab_frame(self.cam))

            hold_until = time.time() + config.POSE_HOLD_SEC
            deadline = (time.time() + config.POSE_HOLD_SEC
                        + config.POSE_CAPTURE_TIMEOUT_SEC)
            # Grab several stable frames at this one pose so a new person
            # is enrolled from "one good picture" with a little natural
            # variation (micro-movements between captures).
            want = max(1, getattr(config, "POSE_FRAMES_PER_POSE", 1))
            got = 0
            stable_since = None
            last_lms = None

            while got < want and time.time() < deadline:
                self.last_loop_at = time.time()      # watchdog heartbeat
                frame = grab_frame(self.cam)
                self.frames.push(frame)
                dets = self.pipe.detect(
                    frame, config.DETECTOR_SCORE_THRESHOLD,
                    config.DETECTOR_NMS_IOU,
                )
                det = largest_detection(dets)
                ok = False
                if det is not None:
                    ok, _ = is_quality_face(det, frame.shape)
                if det is None:
                    status = "step into frame"
                elif not ok:
                    status = "move closer / face the camera"
                elif time.time() < hold_until:
                    status = "hold steady..."
                else:
                    status = f"capturing — hold still ({got + 1}/{want})"
                self.state.patch("register_pose", status=status)

                if ok and time.time() >= hold_until:
                    anchor, eye_dist = landmark_anchor(det)
                    shift_ok = (
                        direction is None
                        or baseline_anchor is None
                        or shift_matches_direction(
                            anchor, baseline_anchor, baseline_eye_dist, direction,
                        )
                    )
                    if shift_ok:
                        if (last_lms is not None
                                and landmarks_drift(det.landmarks, last_lms)
                                <= config.POSE_STABLE_PIXEL_TOL):
                            if stable_since is None:
                                stable_since = time.time()
                            elif time.time() - stable_since >= config.POSE_STABLE_SEC:
                                aligned = align_face(frame, det.landmarks)
                                embeddings.append(self.pipe.embed(aligned))
                                got += 1
                                if direction is None and baseline_anchor is None:
                                    baseline_anchor = anchor
                                    baseline_eye_dist = eye_dist
                                # Re-stabilise before grabbing the next
                                # frame so successive captures aren't the
                                # exact same instant.
                                stable_since = None
                        else:
                            stable_since = None
                        last_lms = det.landmarks
                    else:
                        stable_since = None
                        last_lms = det.landmarks
                else:
                    stable_since = None
                    last_lms = None

            if got == 0:
                self.greeter.say("That's okay, let's give it another try.")
                self.tts.wait_idle(timeout=3)
        self.state.update(register_pose=None)
        return embeddings

    # ---- chat ---------------------------------------------------------

    def _weather_answer(self) -> str | None:
        """One-line current-conditions answer from the weather poller,
        or None if we don't have a fresh reading yet (caller then falls
        back to the LLM)."""
        w = self.state.snapshot().get("weather") or {}
        if not w.get("ok") or w.get("temp_c") is None:
            return None
        temp = round(w["temp_c"])
        label = (w.get("label") or "").strip().lower()
        city = (w.get("city") or "").strip()
        ans = f"It's {temp}°C"
        if label:
            ans += f" and {label}"
        if city:
            ans += f" in {city}"
        ans += " right now."
        humidity = w.get("humidity")
        if humidity is not None:
            ans += f" Humidity is around {humidity}%."
        return ans

    def _project_list_answer(self) -> str | None:
        """Read out the projects currently on display, straight from the
        live list (the admin keeps this current). None if the list is
        empty -- caller then falls through to the LLM."""
        projects = self.state.snapshot().get("projects") or []
        titles = [
            (p.get("title") or "").strip()
            for p in projects if (p.get("title") or "").strip()
        ]
        if not titles:
            return None
        if len(titles) == 1:
            return f"On display today we have {titles[0]}."
        return ("On display today we have "
                + ", ".join(titles[:-1]) + f", and {titles[-1]}.")

    def _solutions_summary(self) -> str | None:
        """List the catalog names for "what tools/solutions were
        developed?". Names only (descriptions would be far too long to read
        out); the visitor can then ask about any one for details."""
        rows = self.db.list_solutions()
        if not rows:
            return None
        names = [(r[1] or "").strip() for r in rows if (r[1] or "").strip()]
        if not names:
            return None
        if len(names) == 1:
            listing = names[0]
        else:
            listing = ", ".join(names[:-1]) + f", and {names[-1]}"
        return (f"We've built {len(names)} solutions: {listing}. Ask me about "
                "any one of them and I'll tell you what it does.")

    def _solution_answer(self, question: str) -> str | None:
        """Match the question against the solutions catalog. Returns a
        spoken answer for a confident match; a catalog-bounded "no match"
        line when the user explicitly said "solution" but nothing fits;
        otherwise None so the caller falls back to the LLM."""
        rows = self.db.list_solutions()
        if not rows:
            return None
        matches = solutions_match.match_solutions(question, rows)
        if solutions_match.is_confident(matches):
            _score, row = matches[0]
            name, desc = row[1], row[2]
            ans = f"Yes -- we've built {name}."
            summary = solutions_match.summarize(desc)
            if summary:
                ans += f" {summary}"
            return ans
        if "solution" in (question or "").lower():
            return ("I don't have a specific solution matching that yet, but "
                    "we've built tools across cloud operations, migration, "
                    "security, FinOps, and app development. Ask about one of "
                    "those and I'll point you to it.")
        return None

    def _submit_chat(self, emp_id: str, question: str):
        # Capture prior conversation turns BEFORE appending this question,
        # so the LLM sees the context but not a duplicate of the current
        # turn. chat.submit() trims to CHAT_HISTORY_TURNS.
        prior_history = list(self.state.snapshot().get("chat_history") or [])
        self._append_chat("user", question)
        # Local-only goodbye detection -- no LLM call. Catches "bye",
        # "thanks bye", "i'm done", etc. (see config.CHAT_GOODBYE_TOKENS).
        # Saves a chat-budget slot AND ~1-2 s of LLM round-trip on the
        # most common session-end gesture.
        if _is_goodbye(question):
            farewell = messages.random_farewell()
            print(f"[chat] goodbye intent matched: {question!r} -> {farewell!r}",
                  flush=True)
            self._append_chat("assistant", farewell)
            self.greeter.say(farewell)
            self._last_chat_at = time.time()
            # Trigger the same IDLE transition as the X button. The
            # camera worker reads idle_event each loop iteration and
            # handles the rest (tts.interrupt, listener.resume, etc).
            self.idle_event.set()
            return
        # Local weather: answer current-conditions questions straight from
        # the weather poller (no LLM, no web search, no budget slot). One
        # crisp line, instant. Falls through to the LLM only if we have no
        # fresh local reading yet.
        if _is_weather_query(question):
            ans = self._weather_answer()
            if ans:
                print(f"[chat] weather intent -> local data: {ans!r}",
                      flush=True)
                self._append_chat("assistant", ans)
                self.greeter.say(ans)
                self._last_chat_at = time.time()
                return
        # Projects on display: answer from the live project list so the
        # kiosk reads out exactly what's there (no LLM, no hallucinated
        # extras, instant). Falls through to the LLM if the list is empty.
        if _is_project_list_query(question):
            ans = self._project_list_answer()
            if ans:
                print(f"[chat] project-list intent -> local data: {ans!r}",
                      flush=True)
                self._append_chat("assistant", ans)
                self.greeter.say(ans)
                self._last_chat_at = time.time()
                return
        # Solutions catalog: "what solutions have you built?" -> summary;
        # "do you have something for X?" -> best catalog match. Searched
        # locally (no LLM); falls through to the LLM only when intent is
        # vague and nothing matches.
        if _is_solution_list_query(question):
            ans = self._solutions_summary()
            if ans:
                print(f"[chat] solution-list intent -> {ans!r}", flush=True)
                self._append_chat("assistant", ans)
                self.greeter.say(ans)
                self._last_chat_at = time.time()
                return
        if _is_solution_query(question):
            ans = self._solution_answer(question)
            if ans:
                print(f"[chat] solution intent -> local match: {ans!r}",
                      flush=True)
                self._append_chat("assistant", ans)
                self.greeter.say(ans)
                self._last_chat_at = time.time()
                return
        if self.chat.budget.remaining(emp_id) <= 0:
            answer = (
                f"Lovely chatting! That's {config.CHAT_MAX_QUESTIONS_PER_SESSION} "
                "questions for now — come say hi again anytime."
            )
            self._append_chat("assistant", answer)
            self.greeter.say(answer)
            return
        self.state.update(chat_pending=True)
        snip = (question[:60] + "...") if len(question) > 63 else question
        print(f"[chat] submit ({self.chat.status()}): {snip!r}", flush=True)
        self._last_chat_at = time.time()
        self._chat_submitted_at = time.time()
        self._chat_future = self.chat.submit(emp_id, question,
                                             history=prior_history)

    def _append_chat(self, role: str, text: str):
        snap = self.state.snapshot()
        history = list(snap.get("chat_history") or [])
        history.append([role, text])
        history = history[-12:]    # cap so SSE diffs stay small
        emp_id = (snap.get("person") or {}).get("emp_id", "anon")
        self.state.update(
            chat_history=history,
            chat_remaining=self.chat.budget.remaining(emp_id),
        )

    # ---- idle dashboard data ------------------------------------------

    def _refresh_idle_data(self):
        # Projects.
        projects = [
            {"id": r[0], "title": r[1], "description": r[2],
             "ordering": r[3]}
            for r in self.db.list_projects()
        ]
        # Sessions: next upcoming.
        nxt = self.db.next_session()
        next_session = None
        if nxt is not None:
            next_session = {
                "id": nxt[0], "title": nxt[1],
                "starts_at": nxt[2], "ends_at": nxt[3], "notes": nxt[4],
            }
        # Metrics.
        best_day, best_count = self.db.interaction_best_day()
        metrics = {
            "total":    self.db.interaction_count_total(),
            "today":    self.db.interaction_count_today(),
            "week":     self.db.interaction_count_this_week(),
            "best_day": best_day,
            "best_count": best_count,
        }
        self.state.update(projects=projects,
                          next_session=next_session,
                          metrics=metrics)

    def _push_metrics(self):
        best_day, best_count = self.db.interaction_best_day()
        self.state.update(metrics={
            "total":    self.db.interaction_count_total(),
            "today":    self.db.interaction_count_today(),
            "week":     self.db.interaction_count_this_week(),
            "best_day": best_day,
            "best_count": best_count,
        })


# ---------- main ----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-register", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    parser.add_argument("--no-display", action="store_true",
                        help="(compat) ignored -- the kiosk has no cv2 window.")
    args = parser.parse_args()

    db = FaceDB()
    state = StateBus()
    state.update(
        brand=getattr(config, "BRAND_NAME", "ECHO SCOPE"),
        wake_phrase=config.WAKE_WORD,
        contact_email=getattr(config, "CONTACT_EMAIL", ""),
    )

    weather = WeatherPoller()
    weather.start()
    state_weather_pump = _start_weather_pump(weather, state)

    fun_facts = FunFactRotator(state)
    fun_facts.start()

    tts = AsyncTTS(make_backend())
    chat = ChatClient(db=db)
    state.update(chat_backend=chat.status(),
                 chat_remaining=config.CHAT_MAX_QUESTIONS_PER_SESSION)

    greeter = Greeter(tts, state, db=db)
    learner = SilentLearner(db)
    frames = FrameStreamer()

    wake_event = threading.Event()
    idle_event = threading.Event()
    register_q: queue.Queue = queue.Queue()
    chat_q: queue.Queue = queue.Queue()

    # Wake-word listener.
    listener: WakeWordListener | None = None
    if not args.no_wake_word:
        try:
            # Print every Vosk hypothesis so it's obvious WHY the wake
            # word isn't firing -- accent / phrasing / mic positioning
            # are usually the cause and were previously invisible.
            _last_partial = [""]

            def _on_partial(text: str):
                if text and text != _last_partial[0]:
                    _last_partial[0] = text
                    print(f"[wake-word] partial: {text!r}", flush=True)

            def _on_final(text: str):
                print(f"[wake-word] FINAL  : {text!r}", flush=True)
                _last_partial[0] = ""

            listener = WakeWordListener(
                config.VOSK_MODEL_DIR, config.WAKE_WORD,
                samplerate=config.WAKE_WORD_SAMPLERATE,
                on_partial=_on_partial,
                on_final=_on_final,
            )
            listener.start()
            # Poll thread: convert listener.is_activated() to wake_event.
            threading.Thread(target=_poll_wake, args=(listener, wake_event),
                              daemon=True).start()
            phrases = ", ".join(repr(p) for p in listener._phrases)
            print(f"[wake-word] listening for any of: {phrases}")
        except WakeWordError as exc:
            print(f"[wake-word] disabled: {exc}")

    # Chat voice.
    chat_voice: ChatVoiceCapture | None = None
    try:
        chat_asr = make_chat_asr()
        chat_voice = ChatVoiceCapture(
            asr=chat_asr,
            out_queue=chat_q,
            wake_listener=listener,
            on_listening_changed=lambda on: state.update(listening=on),
            on_transcribing_changed=lambda on: state.update(transcribing=on),
        )
        print(f"[chat-voice] using {getattr(chat_asr, 'label', 'unknown')}")
        # Warm up the ASR pipeline in a background daemon thread. For
        # Hailo Whisper that's a ~9 s model load (HEF -> NPU + tokenizer
        # files) -- doing it here means the user's first chat-voice
        # tap doesn't eat that cost. The thread runs in this process,
        # so the loaded pipeline sticks around in self._pipeline for
        # future transcribe() calls.
        def _warmup():
            t0 = time.time()
            try:
                chat_asr.warmup()
                print(f"[asr] warmed up in {(time.time() - t0):.1f}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[asr] warmup failed (will retry on first use): "
                       f"{exc!r}", flush=True)
        threading.Thread(target=_warmup, daemon=True,
                         name="asr-warmup").start()
    except ChatVoiceCaptureError as exc:
        print(f"[chat-voice] disabled: {exc}")

    # Worker thread.
    worker = CameraWorker(db, state, frames, tts, chat, greeter, learner,
                          register_q, chat_q,
                          wake_event, idle_event, args, listener=listener)
    worker.start()

    # Watchdog: bounce the process (systemd Restart=always relaunches with
    # clean camera + NPU state) if the worker thread dies or stalls.
    _start_watchdog(worker)
    # Privacy: background retention sweeper (no-op unless DATA_RETENTION_HOURS
    # is set).
    _start_purge_sweeper(db)

    # Flask app -- callbacks bridge the browser to the worker.
    def request_wake():
        wake_event.set()

    def request_idle():
        idle_event.set()

    def request_register(payload):
        register_q.put(payload)

    def request_register_skip():
        register_q.put({"action": "skip"})

    def request_chat(question):
        # Route chat questions through the same chat_q the chat-voice
        # capture uses so the worker only has one entrypoint.
        chat_q.put(question)

    def listen_start():
        if chat_voice is None:
            return
        # Shut up the kiosk if it's mid-utterance -- we shouldn't talk
        # over the user. Drops the pending TTS queue too.
        tts.interrupt()
        chat_voice.start()

    def listen_stop():
        if chat_voice is not None:
            chat_voice.stop()

    app = create_app(state, frames, db,
                     request_wake=request_wake,
                     request_idle=request_idle,
                     request_register=request_register,
                     request_register_skip=request_register_skip,
                     request_chat=request_chat,
                     listen_start=listen_start,
                     listen_stop=listen_stop)

    host = getattr(config, "KIOSK_HOST", "127.0.0.1")
    port = int(os.environ.get("KIOSK_PORT", getattr(config, "KIOSK_PORT", 8080)))
    print(f"echo-scope serving on http://{host}:{port}")
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        # Signal the watchdog/sweeper to stand down so a clean shutdown
        # isn't mistaken for a crash.
        _shutdown.set()
        worker.stop()
        if listener is not None:
            listener.stop()
        weather.stop()
        fun_facts.stop()
        chat.shutdown()
        tts.stop()


_shutdown = threading.Event()


def _start_watchdog(worker: "CameraWorker") -> None:
    """Monitor the camera worker; on death or stall, exit the process so
    the systemd unit (Restart=always) relaunches with clean hardware
    state. A clean restart beats trying to re-acquire leaked camera / NPU
    handles inside a wedged process."""
    def watch():
        while not _shutdown.is_set():
            _shutdown.wait(config.WATCHDOG_POLL_SEC)
            if _shutdown.is_set():
                return
            if not worker.is_alive():
                print("[watchdog] camera worker thread is dead -- exiting "
                      "for restart", flush=True)
                os._exit(1)
            stale = time.time() - getattr(worker, "last_loop_at", time.time())
            if stale > config.WORKER_HEARTBEAT_STALL_SEC:
                print(f"[watchdog] camera worker stalled {stale:.0f}s -- "
                      "exiting for restart", flush=True)
                os._exit(1)
    threading.Thread(target=watch, daemon=True, name="watchdog").start()


def _start_purge_sweeper(db: FaceDB) -> None:
    """Background privacy sweep: delete face data older than the retention
    window. No-op unless config.DATA_RETENTION_HOURS > 0."""
    hours = getattr(config, "DATA_RETENTION_HOURS", 0)
    if hours <= 0:
        return
    interval = max(1, getattr(config, "DATA_PURGE_SWEEP_MIN", 30)) * 60

    def sweep():
        while not _shutdown.is_set():
            try:
                n = db.purge_faces_older_than(hours)
                if n:
                    print(f"[privacy] auto-purged {n} face record(s) older "
                          f"than {hours}h", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[privacy] purge sweep failed: {exc!r}", flush=True)
            _shutdown.wait(interval)
    threading.Thread(target=sweep, daemon=True, name="purge-sweeper").start()
    print(f"[privacy] retention sweeper on: purge faces older than {hours}h "
          f"every {interval // 60}min", flush=True)


def _poll_wake(listener: WakeWordListener, ev: threading.Event):
    while True:
        if listener.is_activated():
            ev.set()
            listener.deactivate()
        time.sleep(0.1)


def _start_weather_pump(weather: WeatherPoller, state: StateBus):
    """Mirror WeatherPoller.get() into state.weather every 30 s."""
    def pump():
        while True:
            w = weather.get()
            state.update(weather={
                "ok": w.get("ok", False),
                "temp_c": w.get("temp_c"),
                "label": w.get("label"),
                "city": w.get("city"),
                "humidity": w.get("humidity"),
                "icon": w.get("icon", "🌡️"),
            })
            time.sleep(30)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    main()
