"""Weather widget for the idle screen.

Background daemon thread that:
  1. Resolves the Pi's location once via https://ipapi.co/json/
     (returns city + lat/lon based on the public IP).
  2. Polls https://api.open-meteo.com/v1/forecast every
     WEATHER_REFRESH_SEC seconds for the current weather there.
  3. Caches the latest result in `_state`. `get()` is safe from any
     thread.

No API key needed for either endpoint.

Failure behaviour: keeps the previous value, sets `stale_for` to how
long ago the last successful fetch was. The renderer can grey it out
when stale.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

import config


# WMO weather codes -> short label. Open-Meteo returns the code in
# `current_weather.weathercode`. https://open-meteo.com/en/docs#weathervariables
_WMO = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
    61: "Rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Freezing rain",
    71: "Snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Showers", 81: "Showers", 82: "Heavy showers",
    85: "Snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


def describe_weather_code(code: int) -> str:
    return _WMO.get(int(code), f"Code {code}")


class WeatherPoller:
    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "ok": False,
            "city": None,
            "temp_c": None,
            "label": None,
            "fetched_at": 0.0,
            "error": None,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get(self) -> dict[str, Any]:
        """Return a copy of the latest state with `stale_for` filled in."""
        with self._lock:
            s = dict(self._state)
        s["stale_for"] = (time.time() - s["fetched_at"]) if s["fetched_at"] else None
        return s

    # internal -----------------------------------------------------------

    def _run(self) -> None:
        lat = lon = city = None
        # Retry geolocation in a loop -- if all providers fail at boot we
        # don't want to give up forever. Back off to WEATHER_REFRESH_SEC
        # between full failures so we don't hammer the providers.
        while not self._stop.is_set() and lat is None:
            lat, lon, city = self._resolve_location()
            if lat is None:
                with self._lock:
                    self._state["error"] = "geo lookup failed"
                self._stop.wait(min(300, config.WEATHER_REFRESH_SEC))
        if self._stop.is_set():
            return
        with self._lock:
            self._state["city"] = city

        while not self._stop.is_set():
            self._poll(lat, lon)
            self._stop.wait(config.WEATHER_REFRESH_SEC)

    def _resolve_location(self) -> tuple[float | None, float | None, str | None]:
        if (config.WEATHER_LATITUDE is not None
                and config.WEATHER_LONGITUDE is not None):
            return (config.WEATHER_LATITUDE, config.WEATHER_LONGITUDE,
                    config.WEATHER_FALLBACK_CITY or "")
        # Three keyless providers tried in order. Each has different
        # failure modes:
        #   ipapi.co     - HTTPS, nicest city names, hard rate limit (429)
        #   ipwho.is     - HTTPS, generous limits, occasionally slow
        #   ip-api.com   - HTTP only on free tier; may be blocked by
        #                  networks that filter outbound port 80
        providers = [
            ("https://ipapi.co/json/",
             lambda j: (float(j["latitude"]), float(j["longitude"]),
                        j.get("city") or "")),
            ("https://ipwho.is/",
             lambda j: (float(j["latitude"]), float(j["longitude"]),
                        j.get("city") or "")),
            ("http://ip-api.com/json/",
             lambda j: (float(j["lat"]), float(j["lon"]),
                        j.get("city") or "")),
        ]
        last_exc = None
        for url, parse in providers:
            try:
                r = requests.get(url, timeout=4)
                r.raise_for_status()
                return parse(r.json())
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[weather] {url} failed: {exc}")
        print(f"[weather] all geo providers failed; last: {last_exc}")
        return None, None, None

    def _poll(self, lat: float, lon: float) -> None:
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "hourly": "relative_humidity_2m",
                },
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            cw = data["current_weather"]
            # Snap the humidity reading to the current_weather.time row
            # (open-meteo's hourly array is aligned by index).
            humidity = None
            try:
                times = data["hourly"]["time"]
                hums = data["hourly"]["relative_humidity_2m"]
                tnow = cw.get("time")
                if tnow in times:
                    humidity = int(hums[times.index(tnow)])
                else:
                    humidity = int(hums[-1])
            except Exception:
                pass
            with self._lock:
                self._state.update({
                    "ok": True,
                    "temp_c": float(cw["temperature"]),
                    "label": describe_weather_code(cw["weathercode"]),
                    "humidity": humidity,
                    "fetched_at": time.time(),
                    "error": None,
                })
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state["error"] = str(exc)
            print(f"[weather] poll failed: {exc}")
