/* ECHO SCOPE kiosk SPA controller.
 *
 * Listens to /events for backend state diffs, mirrors them into the
 * DOM. POSTs to /api/wake, /api/listen/start, /api/chat, /api/register
 * when the user interacts. No frontend framework -- vanilla DOM. */

(() => {
  "use strict";

  const $  = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // Designed for a fixed 1280x800 kiosk frame. If the actual display is
  // smaller, scale the frame down via CSS transform so the layout never
  // wraps or crops.
  function fitScale() {
    const sx = window.innerWidth  / 1280;
    const sy = window.innerHeight / 800;
    const s  = Math.min(sx, sy);
    document.documentElement.style.setProperty("--kiosk-scale", s);
  }
  fitScale();
  window.addEventListener("resize", fitScale);

  // ---- starfield --------------------------------------------------------
  // 100 small cyan dots at random positions twinkling at random rates.
  (function buildStarfield() {
    const sf = document.getElementById("starfield");
    if (!sf) return;
    const N = 100;
    const frag = document.createDocumentFragment();
    for (let i = 0; i < N; i++) {
      const star = document.createElement("div");
      star.className = "star";
      const size = Math.random() * 2 + 1;
      star.style.width  = size + "px";
      star.style.height = size + "px";
      star.style.left   = (Math.random() * 100) + "%";
      star.style.top    = (Math.random() * 100) + "%";
      star.style.animationDuration = (Math.random() * 3 + 2) + "s";
      frag.appendChild(star);
    }
    sf.appendChild(frag);
  })();

  // ---- state mirror ------------------------------------------------------

  let state = {};

  fetch("/api/state")
    .then(r => r.json())
    .then(s => { applyState(s); subscribeEvents(); })
    .catch(err => {
      console.error("initial state fetch failed", err);
      setTimeout(() => location.reload(), 2000);
    });

  setInterval(() => paintClocks(new Date()), 1000);

  function subscribeEvents() {
    const es = new EventSource("/events");
    es.onmessage = (ev) => {
      try {
        const diff = JSON.parse(ev.data);
        if (diff._heartbeat) return;
        applyState(diff);
      } catch (e) {
        console.error("bad SSE payload", e, ev.data);
      }
    };
    es.onerror = () => {
      es.close();
      setTimeout(subscribeEvents, 2000);
    };
  }

  function applyState(diff) {
    Object.assign(state, diff);
    if ("state" in diff) {
      document.body.dataset.state = diff.state;
      $("#idle").hidden   = diff.state !== "IDLE";
      $("#active").hidden = diff.state !== "ACTIVE";
    }
    if ("listening" in diff) {
      document.body.dataset.listening = diff.listening ? "true" : "false";
      bind("listen-label", diff.listening ? "Listening..." : "Tap to speak");
      // Big chat-mic overlay sits over the camera while recording.
      const lo = $("#listen-overlay");
      if (lo) lo.hidden = !diff.listening;
    }
    if ("weather" in diff)        renderWeather(diff.weather);
    if ("metrics" in diff)        renderMetrics(diff.metrics);
    if ("projects" in diff)       renderProjects(diff.projects);
    if ("next_session" in diff)   renderSession(diff.next_session);
    if ("fun_fact" in diff)       renderFunFact(diff.fun_fact);
    if ("toast" in diff)          renderToast(diff.toast);
    if ("wake_phrase" in diff) {
      const phrase = `Hello ${diff.wake_phrase.replace(/^hello\s+/i, "")}!`;
      $$('[data-bind="wake-phrase"]').forEach(e => e.textContent = phrase);
    }
    if ("register_open" in diff)    $("#register-overlay").hidden = !diff.register_open;
    if ("register_message" in diff) bind("register-message", diff.register_message);
    if ("register_pose" in diff)    renderRegisterPose(diff.register_pose);
    if ("chat_history" in diff)     renderChatLog(diff.chat_history);
  }

  function bind(name, value) {
    $$(`[data-bind="${name}"]`).forEach(el => {
      el.textContent = value == null ? "—" : String(value);
    });
  }

  function paintClocks(d) {
    bind("clock-long", d.toLocaleString("en-US", {
      weekday: "short", month: "short", day: "numeric",
      hour: "numeric", minute: "2-digit", hour12: true,
    }));
    bind("clock-short", d.toLocaleString("en-US", {
      hour: "numeric", minute: "2-digit", hour12: true,
    }));
  }

  // ---- weather ----------------------------------------------------------

  function renderWeather(w) {
    if (!w || !w.ok) {
      bind("weather-temp", "—");
      bind("weather-temp-short", "—");
      bind("weather-humidity", "—");
      bind("weather-city", "—");
      return;
    }
    bind("weather-temp", `${Math.round(w.temp_c)}°C`);
    bind("weather-temp-short", `${Math.round(w.temp_c)}°C`);
    bind("weather-humidity", w.humidity != null ? `${w.humidity}%` : "—");
    bind("weather-city", w.city || w.label || "—");
  }

  // ---- metrics ----------------------------------------------------------

  function renderMetrics(m) {
    bind("hi5-today", m.today ?? 0);
    bind("hi5-week",  m.week  ?? 0);
    bind("hi5-best",  m.best_day ? `${m.best_day} (${m.best_count}!)` : "—");
  }

  // ---- projects ---------------------------------------------------------

  function renderProjects(list) {
    const ul = $('[data-bind-list="projects"]');
    if (!ul) return;
    ul.innerHTML = "";
    if (!list || !list.length) {
      const li = document.createElement("li");
      li.className = "placeholder";
      li.textContent = "No projects on display yet.";
      ul.appendChild(li);
      return;
    }
    for (const p of list) {
      const li = document.createElement("li");
      li.textContent = p.title || p[1] || "(untitled)";
      ul.appendChild(li);
    }
  }

  // ---- session ----------------------------------------------------------

  function renderSession(s) {
    const block = document.querySelector('[data-bind-block="session"]');
    const placeholder = document.querySelector(
      '[data-bind-show-if-empty="session"]'
    );
    if (!s) {
      if (block) block.style.display = "none";
      if (placeholder) placeholder.style.display = "";
      return;
    }
    if (block) block.style.display = "";
    if (placeholder) placeholder.style.display = "none";
    bind("session-title", s.title);
    bind("session-when", formatSessionWhen(s));
  }
  function formatSessionWhen(s) {
    try {
      const start = new Date(s.starts_at.replace(" ", "T"));
      const dateStr = start.toLocaleDateString("en-US", {
        weekday: "short", month: "short", day: "numeric",
      });
      const timeStr = start.toLocaleTimeString("en-US", {
        hour: "numeric", minute: "2-digit", hour12: true,
      });
      let end = "";
      if (s.ends_at) {
        const e = new Date(s.ends_at.replace(" ", "T"));
        end = "-" + e.toLocaleTimeString("en-US", {
          hour: "numeric", minute: "2-digit", hour12: true,
        });
      }
      return `${dateStr} • ${timeStr}${end}`;
    } catch (e) {
      return s.starts_at;
    }
  }

  // ---- fun fact (rotates between FACT and TIP from backend) -------------

  function renderFunFact(f) {
    if (!f) return;
    bind("fact-icon", f.icon || "💡");
    bind("fact-type", f.type || "AI FUN FACT");
    bind("fact-content", f.content || "");
    const card = document.querySelector(".fun-fact");
    if (card) {
      const kind = (f.type || "").toLowerCase().includes("tip") ? "tip" : "fact";
      card.dataset.factKind = kind;
      // Replay the fade-in animation.
      card.style.animation = "none";
      void card.offsetWidth;
      card.style.animation = "";
    }
  }

  // ---- toast ------------------------------------------------------------

  let toastTimer = null;
  function renderToast(t) {
    const el = $("#toast");
    if (!el) return;
    if (!t || !t.text) { el.hidden = true; return; }
    el.textContent = t.text;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 4000);
  }

  // ---- register pose ----------------------------------------------------
  // Pose-capture progress lives on the active screen now (not inside
  // the registration form overlay) so the user can see the camera feed
  // while turning their head.
  function renderRegisterPose(p) {
    const box = $("#active-pose");
    if (!box) return;
    if (!p) { box.hidden = true; return; }
    box.hidden = false;
    $("#pose-idx").textContent   = p.idx   ?? "?";
    $("#pose-total").textContent = p.total ?? "?";
    $("#pose-prompt").textContent = p.prompt ?? "—";
    $("#pose-status").textContent = p.status ?? "";
  }

  // ---- chat log ---------------------------------------------------------

  function renderChatLog(history) {
    const box = $("#chat-log");
    if (!box || !Array.isArray(history)) return;
    box.innerHTML = "";
    for (const [role, text] of history) {
      const row = document.createElement("div");
      row.className = "row " + role;
      row.textContent = text;
      box.appendChild(row);
    }
    box.scrollTop = box.scrollHeight;
  }

  // ---- user input -------------------------------------------------------

  // IDLE: tap anywhere wakes the kiosk.
  document.getElementById("idle").addEventListener("click", () => {
    if (state.state === "IDLE") post("/api/wake");
  });

  document.getElementById("close-active").addEventListener("click", (ev) => {
    ev.stopPropagation();
    // Optimistically flip locally; the backend's idle timeout sends
    // an authoritative state update soon after.
    document.body.dataset.state = "IDLE";
    $("#idle").hidden   = false;
    $("#active").hidden = true;
  });

  document.getElementById("listen-toggle").addEventListener("click", () => {
    if (state.listening) post("/api/listen/stop");
    else                 post("/api/listen/start");
  });
  // Tap anywhere on the listen overlay to stop listening.
  $("#listen-overlay")?.addEventListener("click", () => {
    if (state.listening) post("/api/listen/stop");
  });

  const chatForm = $("#chat-form");
  if (chatForm) {
    chatForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const input = $("#chat-input");
      const q = input.value.trim();
      if (!q) return;
      input.value = "";
      postJSON("/api/chat", { question: q });
    });
  }
  $("#chat-close")?.addEventListener("click", () => {
    $("#chat-overlay").hidden = true;
  });

  $("#register-form")?.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const emp_id = $("#register-emp-id").value.trim();
    const name   = $("#register-name").value.trim();
    if (!emp_id || !name) return;
    postJSON("/api/register", { emp_id, name });
  });
  $("#register-close")?.addEventListener("click", () => {
    $("#register-overlay").hidden = true;
  });

  function post(url) {
    fetch(url, { method: "POST" }).catch(e => console.warn("POST failed", url, e));
  }
  function postJSON(url, body) {
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(e => console.warn("POST failed", url, e));
  }

  // ---- MJPEG: only hold the socket while we're ACTIVE ------------------

  const camImg = $("#camera-stream");
  let camOn = false;
  setInterval(() => {
    if (state.state === "ACTIVE" && !camOn) {
      camImg.src = "/camera.mjpg?ts=" + Date.now();
      camOn = true;
      document.body.dataset.cameraOn = "true";
    } else if (state.state === "IDLE" && camOn) {
      camImg.src = "";
      camOn = false;
      document.body.dataset.cameraOn = "false";
    }
  }, 500);
})();
