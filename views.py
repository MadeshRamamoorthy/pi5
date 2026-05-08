"""Rendering helpers for the kiosk UI.

Two top-level functions:
  render_idle(weather, counter, theme, size)       -> BGR image
  render_active(camera_frame, panel_image)         -> hstacked BGR image

Right-panel renderers (used by the active-state RightPanel state machine):
  render_projects_panel(projects, selected_idx, scroll, size)
  render_register_panel(form_state, message, size)
  render_chat_panel(history, remaining, input_mode, partial_text,
                     backend_label, size)

Everything draws onto numpy uint8 BGR canvases via cv2.putText / cv2.rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------- idle screen ----------------


def render_idle(weather: dict, counter: int, size: tuple[int, int]) -> np.ndarray:
    """size = (width, height)."""
    w, h = size
    bg = config.THEME["bg"]
    fg = config.THEME["fg"]
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)

    # Centered welcome text -- two lines for legibility.
    t1 = "Welcome to"
    t2 = "Echo AI"
    s1, s2 = 1.6, 3.2
    (tw1, th1), _ = cv2.getTextSize(t1, FONT, s1, 3)
    (tw2, th2), _ = cv2.getTextSize(t2, FONT, s2, 6)
    cy = h // 2 - th2 // 2
    cv2.putText(canvas, t1, ((w - tw1) // 2, cy - th2 // 2 - 20),
                FONT, s1, fg, 3, cv2.LINE_AA)
    cv2.putText(canvas, t2, ((w - tw2) // 2, cy + th2 // 2 + 30),
                FONT, s2, fg, 6, cv2.LINE_AA)

    # Sub-instruction.
    hint = f"Say '{config.WAKE_WORD}' to begin"
    (thw, thh), _ = cv2.getTextSize(hint, FONT, 0.8, 2)
    cv2.putText(canvas, hint, ((w - thw) // 2, cy + th2 // 2 + 90),
                FONT, 0.8, fg, 2, cv2.LINE_AA)

    # Top-right weather widget.
    _draw_weather(canvas, weather, w)

    # Bottom-right interaction counter.
    _draw_counter(canvas, counter, w, h)

    return canvas


def _draw_weather(canvas, weather: dict, width: int) -> None:
    fg = config.THEME["fg"]
    if not weather or not weather.get("ok"):
        text = "weather: —"
        if weather and weather.get("error"):
            text = "weather: offline"
    else:
        temp = weather.get("temp_c")
        label = weather.get("label") or ""
        city = weather.get("city") or ""
        text = f"{label}  {temp:.0f}°C"
        if city:
            text = f"{city}  ·  {text}"
        stale = weather.get("stale_for")
        if stale and stale > 2 * config.WEATHER_REFRESH_SEC:
            text += "  (stale)"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    pad = 10
    x = width - tw - 24
    y = 28 + th
    cv2.putText(canvas, text, (x, y), FONT, 0.7, fg, 2, cv2.LINE_AA)


def _draw_counter(canvas, counter: int, width: int, height: int) -> None:
    fg = config.THEME["fg"]
    big = f"{counter}"
    small = "interactions"
    (bw, bh), _ = cv2.getTextSize(big, FONT, 1.4, 3)
    (sw, sh), _ = cv2.getTextSize(small, FONT, 0.6, 1)
    x_right = width - 24
    y_big = height - 30
    y_small = y_big - bh - 4
    cv2.putText(canvas, small, (x_right - sw, y_small),
                FONT, 0.6, fg, 1, cv2.LINE_AA)
    cv2.putText(canvas, big, (x_right - bw, y_big),
                FONT, 1.4, fg, 3, cv2.LINE_AA)


# ---------------- active screen ----------------


def render_active(camera_frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    if panel.shape[0] != camera_frame.shape[0]:
        panel = cv2.resize(
            panel, (panel.shape[1], camera_frame.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.hstack([camera_frame, panel])


# ---------------- right-panel: shared chrome ----------------


def _panel_canvas(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    canvas = np.full((h, w, 3), (24, 24, 24), dtype=np.uint8)
    return canvas


def _draw_tabs(canvas: np.ndarray, active_tab: str,
               session_count: int) -> int:
    """Returns the y-coord at which body content can start."""
    w = canvas.shape[1]
    tabs = ["PROJECTS", "REGISTER", "CHAT"]
    tab_w = w // len(tabs)
    y = 38
    for i, name in enumerate(tabs):
        x0 = i * tab_w
        x1 = x0 + tab_w
        if name == active_tab:
            cv2.rectangle(canvas, (x0, 0), (x1, y), (52, 52, 52), -1)
            colour = (240, 240, 240)
        else:
            colour = (140, 140, 140)
        (tw, th), _ = cv2.getTextSize(name, FONT, 0.55, 1)
        cv2.putText(canvas, name, (x0 + (tab_w - tw) // 2, 24),
                    FONT, 0.55, colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, y), (w, y), (60, 60, 60), 1)
    # Counter bottom-right of the tab strip.
    badge = f"this session: {session_count}"
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.42, 1)
    cv2.putText(canvas, badge, (w - bw - 8, y + 14),
                FONT, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    return y + 28


def _wrap(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    out = []
    while text:
        if len(text) <= max_chars:
            out.append(text)
            break
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out


# ---------------- right-panel: projects ----------------


def render_projects_panel(projects: list, selected_idx: int, scroll: int,
                          session_count: int,
                          size: tuple[int, int]) -> np.ndarray:
    canvas = _panel_canvas(size)
    y = _draw_tabs(canvas, "PROJECTS", session_count)

    if not projects:
        cv2.putText(canvas, "(no projects yet)", (16, y + 20),
                    FONT, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, "open the admin UI to add some.",
                    (16, y + 46), FONT, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        return canvas

    chars = max(20, (size[0] - 32) // 8)
    for i, (pid, title, desc, _ordering, _created) in enumerate(
        projects[scroll:], start=scroll
    ):
        if y > size[1] - 40:
            break
        is_sel = (i == selected_idx)
        if is_sel:
            cv2.rectangle(canvas, (4, y - 16), (size[0] - 4, y + 8),
                          (40, 60, 90), -1)
        head_colour = (240, 230, 140) if is_sel else (200, 200, 200)
        cv2.putText(canvas, f"{i + 1:>2}. {title}", (12, y),
                    FONT, 0.6, head_colour, 1, cv2.LINE_AA)
        y += 22
        for line in _wrap(desc or "", chars):
            cv2.putText(canvas, line, (28, y),
                        FONT, 0.46, (170, 170, 170), 1, cv2.LINE_AA)
            y += 18
            if y > size[1] - 40:
                break
        y += 6

    cv2.putText(canvas, "P projects   R register   C chat",
                (12, size[1] - 14), FONT, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
    return canvas


# ---------------- right-panel: register ----------------


@dataclass
class RegisterFormState:
    emp_id: str = ""
    name: str = ""
    focused: str = "emp_id"   # "emp_id" or "name"
    busy: bool = False
    captured_first_emb: bool = False


def render_register_panel(form: RegisterFormState, message: str,
                          session_count: int,
                          size: tuple[int, int]) -> np.ndarray:
    canvas = _panel_canvas(size)
    y = _draw_tabs(canvas, "REGISTER", session_count)

    cv2.putText(canvas, "Register a new face", (16, y + 16),
                FONT, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    y += 50

    def field(label: str, value: str, focused: bool):
        nonlocal y
        cv2.putText(canvas, label, (16, y), FONT, 0.5, (160, 160, 160),
                    1, cv2.LINE_AA)
        y += 22
        box_top = y - 18
        box_bot = y + 12
        col = (200, 200, 200) if focused else (80, 80, 80)
        cv2.rectangle(canvas, (16, box_top), (size[0] - 16, box_bot), col, 1)
        text = value + ("|" if focused else "")
        cv2.putText(canvas, text, (24, y),
                    FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        y += 36

    field("Employee ID", form.emp_id, form.focused == "emp_id")
    field("Name",        form.name,   form.focused == "name")

    cv2.putText(canvas,
                "Tab: switch field   Enter: submit   Esc: cancel",
                (16, y + 8), FONT, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    if message:
        for line in _wrap(message, max(20, (size[0] - 32) // 8)):
            y += 24
            cv2.putText(canvas, line, (16, y), FONT, 0.5,
                        (140, 220, 140) if form.busy else (220, 200, 100),
                        1, cv2.LINE_AA)
    return canvas


# ---------------- right-panel: chat ----------------


def render_chat_panel(history: list, remaining: int, input_mode: str,
                      partial_text: str, backend_label: str,
                      session_count: int,
                      size: tuple[int, int]) -> np.ndarray:
    canvas = _panel_canvas(size)
    y = _draw_tabs(canvas, "CHAT", session_count)

    head = f"{backend_label}   ·   Q remaining: {remaining}/{config.CHAT_MAX_QUESTIONS_PER_SESSION}"
    cv2.putText(canvas, head, (16, y + 14),
                FONT, 0.5, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"input: {input_mode.upper()}  (V to toggle)",
                (16, y + 36), FONT, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    y += 56
    cv2.line(canvas, (10, y), (size[0] - 10, y), (50, 50, 50), 1)
    y += 18

    chars = max(20, (size[0] - 32) // 8)
    for role, text in history[-8:]:
        prefix, colour = (("You",  (240, 230, 140))
                          if role == "user"
                          else ("Echo", (180, 220, 255)))
        cv2.putText(canvas, prefix, (16, y),
                    FONT, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        y += 18
        for line in _wrap(text, chars):
            cv2.putText(canvas, line, (28, y),
                        FONT, 0.5, colour, 1, cv2.LINE_AA)
            y += 22
            if y > size[1] - 70:
                break
        y += 6
        if y > size[1] - 70:
            break

    # Bottom: input field / partial speech.
    box_top = size[1] - 60
    box_bot = size[1] - 18
    cv2.rectangle(canvas, (12, box_top), (size[0] - 12, box_bot),
                  (80, 80, 80), 1)
    placeholder = ("type and press Enter..." if input_mode == "keyboard"
                   else "speak your question...")
    text = partial_text or placeholder
    text_col = (240, 240, 240) if partial_text else (110, 110, 110)
    if input_mode == "keyboard" and partial_text:
        text = partial_text + "|"
    cv2.putText(canvas, text[:chars], (22, box_bot - 14),
                FONT, 0.5, text_col, 1, cv2.LINE_AA)
    return canvas
