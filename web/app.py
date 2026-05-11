"""Flask app: serves the SPA, the MJPEG camera stream, the SSE state
feed, and a handful of POST endpoints the browser uses to drive the
kiosk (wake / listen / chat / register / sessions / projects)."""

from __future__ import annotations

import os
import time
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
    request_register: Callable[[dict], None],
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
        )

    @app.route("/admin")
    def admin():
        if not _check_admin_auth():
            return _request_admin_auth()
        return render_template("admin.html")

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

    # -------- projects (admin) -------------------------------------------

    @app.route("/api/projects", methods=["GET"])
    def api_projects_list():
        rows = db.list_projects()
        return jsonify([
            {"id": r[0], "title": r[1], "description": r[2],
             "ordering": r[3], "created_at": r[4]} for r in rows
        ])

    @app.route("/api/projects", methods=["POST"])
    def api_projects_create():
        if not _check_admin_auth():
            return _request_admin_auth()
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        if not title:
            return ("title required", 400)
        desc = (body.get("description") or "").strip()
        try:
            ordering = int(body.get("ordering", 0))
        except (TypeError, ValueError):
            ordering = 0
        pid = db.add_project(title, desc, ordering)
        return jsonify({"id": pid})

    @app.route("/api/projects/<int:pid>", methods=["POST", "DELETE"])
    def api_projects_edit(pid):
        if not _check_admin_auth():
            return _request_admin_auth()
        if request.method == "DELETE":
            if not db.delete_project(pid):
                return ("not found", 404)
            return ("", 204)
        body = request.get_json(silent=True) or {}
        kw = {}
        for key in ("title", "description"):
            if key in body:
                kw[key] = (body[key] or "").strip()
        if "ordering" in body:
            try:
                kw["ordering"] = int(body["ordering"])
            except (TypeError, ValueError):
                pass
        if not kw or not db.update_project(pid, **kw):
            return ("no change", 400)
        return ("", 204)

    # -------- sessions (admin) -------------------------------------------

    @app.route("/api/sessions", methods=["GET"])
    def api_sessions_list():
        rows = db.list_sessions(upcoming_only=False)
        return jsonify([
            {"id": r[0], "title": r[1], "starts_at": r[2], "ends_at": r[3],
             "notes": r[4]} for r in rows
        ])

    @app.route("/api/sessions", methods=["POST"])
    def api_sessions_create():
        if not _check_admin_auth():
            return _request_admin_auth()
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        starts_at = (body.get("starts_at") or "").strip()
        if not title or not starts_at:
            return ("title and starts_at required", 400)
        sid = db.add_session(
            title, starts_at,
            body.get("ends_at") or None,
            (body.get("notes") or "").strip(),
        )
        return jsonify({"id": sid})

    @app.route("/api/sessions/<int:sid>", methods=["POST", "DELETE"])
    def api_sessions_edit(sid):
        if not _check_admin_auth():
            return _request_admin_auth()
        if request.method == "DELETE":
            if not db.delete_session(sid):
                return ("not found", 404)
            return ("", 204)
        body = request.get_json(silent=True) or {}
        ok = db.update_session(sid, **{
            k: (body[k] or "").strip() if isinstance(body.get(k), str) else body.get(k)
            for k in ("title", "starts_at", "ends_at", "notes") if k in body
        })
        return ("", 204) if ok else ("no change", 400)

    # -------- metrics ----------------------------------------------------

    @app.route("/api/metrics")
    def api_metrics():
        best_day, best_count = db.interaction_best_day()
        return jsonify({
            "total": db.interaction_count_total(),
            "today": db.interaction_count_today(),
            "week": db.interaction_count_this_week(),
            "best_day": best_day,
            "best_count": best_count,
        })

    return app


# -------- HTTP basic auth helpers ------------------------------------------


def _check_admin_auth() -> bool:
    user = os.environ.get("KIOSK_ADMIN_USER")
    pw = os.environ.get("KIOSK_ADMIN_PASS")
    # If unset, admin is open -- fine for kiosk-local LAN. Set both env
    # vars in production deployments.
    if not user or not pw:
        return True
    auth = request.authorization
    return bool(auth and auth.username == user and auth.password == pw)


def _request_admin_auth():
    return Response(
        "Authentication required", 401,
        {"WWW-Authenticate": 'Basic realm="ECHO SCOPE admin"'},
    )
