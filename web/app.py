"""Flask app: serves the kiosk SPA, the MJPEG camera stream, the SSE
state feed, and the POST endpoints the browser uses to drive the
kiosk (wake / listen / chat / register).

Admin routes (projects / sessions / employees CRUD, /admin HTML,
photo register) live in `web.admin_app` and are served by a separate
process. See docs/admin.md.
"""

from __future__ import annotations

from typing import Callable

from flask import (Flask, Response, jsonify, render_template, request,
                   stream_with_context)

import config
from state import StateBus, encode_sse


def create_app(
    state: StateBus,
    frames,                       # FrameStreamer
    db,                           # FaceDB
    request_wake: Callable[[], None],
    request_idle: Callable[[], None],
    request_register: Callable[[dict], None],
    request_register_skip: Callable[[], None],
    request_chat: Callable[[str], None],
    listen_start: Callable[[], None],
    listen_stop: Callable[[], None],
) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # -------- HTML --------------------------------------------------------

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            brand=getattr(config, "BRAND_NAME", "ECHO SCOPE"),
            wake_phrase=getattr(config, "WAKE_WORD", "hello echo scope"),
            contact=getattr(config, "CONTACT_EMAIL", ""),
            game_enabled=bool(getattr(config, "EMOTION_GAME_ENABLED", False)),
        )

    # -------- state feed --------------------------------------------------

    @app.route("/api/state")
    def api_state():
        return jsonify(state.snapshot())

    @app.route("/events")
    def events():
        sub = state.subscribe()

        @stream_with_context
        def gen():
            try:
                for diff in sub.stream():
                    yield encode_sse(diff)
            finally:
                sub.close()

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # -------- camera stream ---------------------------------------------

    @app.route("/camera.mjpg")
    def camera_mjpg():
        return Response(
            frames.generator(),
            mimetype="multipart/x-mixed-replace; boundary=echoframe",
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
        )

    # -------- actions -----------------------------------------------------

    @app.route("/api/wake", methods=["POST"])
    def api_wake():
        request_wake()
        return ("", 204)

    @app.route("/api/idle", methods=["POST"])
    def api_idle():
        """User tapped the close (X) button on the camera screen. Force
        the kiosk back to IDLE immediately rather than waiting for the
        idle timeout."""
        request_idle()
        return ("", 204)

    @app.route("/api/listen/start", methods=["POST"])
    def api_listen_start():
        listen_start()
        return ("", 204)

    @app.route("/api/listen/stop", methods=["POST"])
    def api_listen_stop():
        listen_stop()
        return ("", 204)

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        if not question:
            return ("question required", 400)
        request_chat(question)
        return ("", 204)

    @app.route("/api/register", methods=["POST"])
    def api_register():
        body = request.get_json(silent=True) or {}
        emp_id = (body.get("emp_id") or "").strip()
        name = (body.get("name") or "").strip()
        if not emp_id or not name:
            return ("emp_id and name required", 400)
        request_register({"emp_id": emp_id, "name": name})
        return ("", 204)

    @app.route("/api/register/skip", methods=["POST"])
    def api_register_skip():
        """User declined the auto-register prompt. Close the overlay and
        cool down the auto-pop trigger so we don't immediately re-open."""
        request_register_skip()
        return ("", 204)

    # -------- facial-expression game (AWS Rekognition) --------------------
    # Lazily create the emotion client the first time /api/game/snap is
    # hit, so the app boots even when AWS creds are missing.
    game_client = {"c": None}

    @app.route("/api/game/snap", methods=["POST"])
    def api_game_snap():
        """Body: multipart form field 'frame' = JPEG bytes of the current
        camera frame captured client-side. Returns Rekognition's top
        emotion + full confidence map for the game overlay to render."""
        if not getattr(config, "EMOTION_GAME_ENABLED", False):
            return jsonify({"ok": False, "error": "game disabled"}), 403
        f = request.files.get("frame")
        if not f:
            return jsonify({"ok": False, "error": "no frame in request"}), 400
        blob = f.read()
        if not blob:
            return jsonify({"ok": False, "error": "empty frame"}), 400
        if game_client["c"] is None:
            from emotion_cloud import make_client
            game_client["c"] = make_client()
            if game_client["c"] is None:
                return jsonify({"ok": False,
                                "error": "emotion client unavailable"}), 503
        return jsonify(game_client["c"].detect(blob))

    return app
