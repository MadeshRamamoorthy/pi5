"""AWS Rekognition emotion detection for the facial-expression game.

Thin wrapper around Rekognition's DetectFaces + emotion attribute. Called
from the /api/game/snap Flask endpoint. Kept isolated from the recognition
pipeline so it's easy to disable (EMOTION_GAME_ENABLED=false) or swap
providers later.

AWS returns 8 emotions per face: HAPPY, SAD, ANGRY, CONFUSED, DISGUSTED,
SURPRISED, CALM, FEAR. We pick the biggest face in the frame (consistent
with the rest of the app) and return the top emotion plus the full
confidence map for the UI to show close runners-up.
"""

from __future__ import annotations

import os
import threading


class EmotionCloudError(RuntimeError):
    pass


class AwsEmotionClient:
    """boto3 Rekognition client, initialised lazily on first use so the
    Flask app boots even when AWS creds aren't set (game just fails
    cleanly at snap time)."""

    def __init__(self, region: str):
        self._region = region
        self._client = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            try:
                import boto3  # noqa: F401
            except ImportError as exc:
                raise EmotionCloudError(
                    "boto3 not installed. Run: pip install boto3"
                ) from exc
            import boto3
            self._client = boto3.client(
                "rekognition", region_name=self._region,
            )

    def detect(self, jpeg_bytes: bytes) -> dict:
        """Detect emotions on the biggest face in `jpeg_bytes`.

        Returns:
            {
                "ok": True,
                "top": "HAPPY",
                "confidence": 87.2,          # 0-100 for the top emotion
                "all": {"HAPPY": 87.2, ...}, # every emotion, sorted desc
                "faces": 1,
            }
            or {"ok": False, "error": "..."} on any failure.
        """
        if not jpeg_bytes:
            return {"ok": False, "error": "empty frame"}
        try:
            self._ensure()
            resp = self._client.detect_faces(
                Image={"Bytes": jpeg_bytes},
                Attributes=["ALL"],
            )
        except EmotionCloudError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # boto3 ClientError, network, etc.
            return {"ok": False,
                    "error": f"AWS Rekognition failed: {exc!r}"}
        faces = resp.get("FaceDetails", []) or []
        if not faces:
            return {"ok": False, "error": "no face detected", "faces": 0}
        # Pick the biggest face (consistent with the recognition pipeline).
        faces.sort(
            key=lambda f: (f.get("BoundingBox", {}).get("Width", 0)
                           * f.get("BoundingBox", {}).get("Height", 0)),
            reverse=True,
        )
        emotions = list(faces[0].get("Emotions", []) or [])
        if not emotions:
            return {"ok": False, "error": "no emotion attributes returned",
                    "faces": len(faces)}
        emotions.sort(key=lambda e: e.get("Confidence", 0), reverse=True)
        top = emotions[0]
        return {
            "ok": True,
            "top": top.get("Type", ""),
            "confidence": float(top.get("Confidence", 0.0)),
            "all": {e["Type"]: float(e["Confidence"]) for e in emotions},
            "faces": len(faces),
        }


def make_client() -> "AwsEmotionClient | None":
    """Factory. Returns None when the feature is disabled via config so
    callers can no-op without importing boto3."""
    import config
    if not getattr(config, "EMOTION_GAME_ENABLED", False):
        return None
    region = (os.environ.get("AWS_REGION")
              or getattr(config, "AWS_REGION", "ca-central-1"))
    return AwsEmotionClient(region)
