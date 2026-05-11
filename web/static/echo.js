/* ECHO SCOPE kiosk SPA controller.
 *
 * Listens to /events for backend state diffs, mirrors them into the
 * DOM. POSTs to /api/wake, /api/listen/start, /api/chat, /api/register
 * when the user interacts. No frontend framework -- vanilla DOM. */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // Local mirror of the backend state. Seeded by /api/state on load
  // and updated by /events diffs.
  let state = {};

  // ============================== bootstrap

  fetch("/api/state")
    .then(r => r.json())
    .then(s => { applyState(s); subscribeEvents(); })
    .catch(err => {
      console.error("initial state fetch failed", err);
      // Retry in 2s
      setTimeout(() => location.reload(), 2000);
    });

  // Local clock tick (server also pushes now_epoch every few s, but
  // ticking locally keeps the display smooth).
  setInterval(() => paintClocks(new Date()), 1000);

  // ============================== SSE

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
    es.onerror = (e) => {
      console.warn("SSE error -- reconnecting in 2s", e);
      es.close();
      setTimeout(subscribeEvents, 2000);
    };
  }

  // ============================== state -> DOM

  function applyState(diff) {
    Object.assign(state, diff);
    if ("state" in diff) {
      document.body.dataset.state = diff.state;
      $("#idle").hidden = diff.state !== "IDLE";
      $("#active").hidden = diff.state !== "ACTIVE";
    }
    if ("listening" in diff) {
      document.body.dataset.listening = diff.listening ? "true" : "false";
    }
    if ("weather" in diff)        renderWeather(diff.weather);
    if ("metrics" in diff)        renderMetrics(diff.metrics);
    if ("projects" in diff)       renderProjects(diff.projects);
    if ("next_session" in diff)   renderSession(diff.next_session);
    if ("fun_fact" in diff)       renderFunFact(diff.fun_fact);
    if ("toast" in diff)          renderToast(diff.toast);
    if ("wake_phrase" in diff)    $$('[data-bind="wake-phrase"]').forEach(e => e.textContent = `Hello ${diff.wake_phrase.replace(/^hello\s+/i, "")}!`);
    if ("register_open" in diff)  $("#register-overlay").hidden = !diff.register_open;
    if ("register_message" in diff) bind("register-message", diff.register_message);
    if ("register_pose" in diff)  renderRegisterPose(diff.register_pose);
    if ("chat_history" in diff)   renderChatLog(diff.chat_history);
    if ("listening" in diff)      bind("listen-label", diff.listening ? "Listening..." : "Tap to speak");
  }

  function bind(name, value) {
    $$(`[data-bind="${name}"]`).forEach(el => {
      el.textContent = value == null ? "—" : String(value);
    });
  }

  function paintClocks(d) {
    const opts = { weekday: "short", month: "short", day: "numeric",
                   hour: "numeric", minute: "2-digit", hour12: true };
    bind("clock-long", d.toLocaleString("en-US", opts));
    bind("clock-short", d.toLocaleString("en-US", {
      hour: "numeric", minute: "2-digit", hour12: true,
    }));
  }

  // ---- weather

  const WEATHER_ICONS = {
    "Clear": "☀️", "Mostly clear": "🌤️", "Partly cloudy": "⛅",
    "Overcast": "☁️", "Fog": "🌫", "Rime fog": "🌫",
    "Drizzle": "🌦️", "Rain": "🌧️", "Heavy rain": "⛈",
    "Showers": "🌦️", "Heavy showers": "⛈",
    "Snow": "❄️", "Heavy snow": "❄️", "Snow grains": "❄️",
    "Thunderstorm": "⛈", "Freezing rain": "🌨️",
  };
  function renderWeather(w) {
    if (!w || !w.ok) {
      bind("weather-icon", "❓");
      bind("weather-temp", "—");
      bind("weather-temp-short", "—");
      bind("weather-humidity", "—");
      return;
    }
    bind("weather-icon", WEATHER_ICONS[w.label] || "🌡️");
    bind("weather-temp", `${Math.round(w.temp_c)}°C`);
    bind("weather-temp-short", `${Math.round(w.temp_c)}°C`);
    bind("weather-humidity", w.humidity != null ? `${w.humidity}%` : "—");
  }

  // ---- metrics (hi-5s)

  function renderMetrics(m) {
    bind("hi5-today", m.today ?? 0);
    bind("hi5-week",  `${m.week ?? 0}`);
    bind("hi5-best",  m.best_day ? `${m.best_day} (${m.best_count}!)` : "—");
  }

  // ---- projects

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

  // ---- session

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
        end = " – " + e.toLocaleTimeString("en-US", {
          hour: "numeric", minute: "2-digit", hour12: true,
        });
      }
      return `${dateStr} • ${timeStr}${end}`;
    } catch (e) {
      return s.starts_at;
    }
  }

  // ---- fun fact

  function renderFunFact(f) {
    if (!f) return;
    bind("fact-icon", f.icon || "💡");
    bind("fact-type", f.type || "AI FUN FACT");
    bind("fact-content", f.content || "");
  }

  // ---- toast

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

  // ---- register pose progress

  function renderRegisterPose(p) {
    const box = $("#register-pose");
    if (!box) return;
    if (!p) { box.hidden = true; return; }
    box.hidden = false;
    $("#pose-idx").textContent = p.idx ?? "?";
    $("#pose-total").textContent = p.total ?? "?";
    $("#pose-prompt").textContent = p.prompt ?? "—";
    $("#pose-status").textContent = p.status ?? "";
  }

  // ---- chat log

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

  // ============================== user input

  // IDLE: tap anywhere wakes the kiosk.
  document.getElementById("idle").addEventListener("click", () => {
    if (state.state === "IDLE") post("/api/wake");
  });

  // Active close (X).
  document.getElementById("close-active").addEventListener("click", (ev) => {
    ev.stopPropagation();
    // No /api/idle endpoint -- the backend drops to idle after the
    // configured timeout. Hide active locally so the user sees an
    // immediate response; a state push will follow.
    document.body.dataset.state = "IDLE";
    $("#idle").hidden = false;
    $("#active").hidden = true;
  });

  // Listening pill.
  document.getElementById("listen-toggle").addEventListener("click", () => {
    if (state.listening) post("/api/listen/stop");
    else                 post("/api/listen/start");
  });

  // Chat overlay
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

  // Register overlay
  $("#register-form")?.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const emp_id = $("#register-emp-id").value.trim();
    const name = $("#register-name").value.trim();
    if (!emp_id || !name) return;
    postJSON("/api/register", { emp_id, name });
  });
  $("#register-close")?.addEventListener("click", () => {
    $("#register-overlay").hidden = true;
  });

  // ============================== helpers

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

  // Once we're active, point the camera <img> at the MJPEG endpoint.
  // We delay this so the browser doesn't keep an MJPEG socket open
  // while idle.
  const camImg = $("#camera-stream");
  let camOn = false;
  setInterval(() => {
    if (state.state === "ACTIVE" && !camOn) {
      camImg.src = "/camera.mjpg?ts=" + Date.now();
      camOn = true;
    } else if (state.state === "IDLE" && camOn) {
      camImg.src = "";
      camOn = false;
    }
  }, 500);
})();
