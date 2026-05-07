"""Hailo-10H inference wrappers for face detection (SCRFD) and embedding (ArcFace).

Uses HailoRT's Python API. The HEF files are produced from the Hailo Model Zoo
(see README). This module assumes single-stream synchronous inference, which is
plenty for a webcam-rate application on a Pi 5 + Hailo-10H.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2) in original image pixels
    score: float
    landmarks: np.ndarray  # (5, 2)


class _HailoModel:
    """Minimal single-network HEF runner."""

    def __init__(self, hef_path: Path, vdevice: VDevice):
        self.hef = HEF(str(hef_path))
        cfg = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = vdevice.configure(self.hef, cfg)[0]
        self.network_params = self.network_group.create_params()
        self.input_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        info = self.hef.get_input_vstream_infos()[0]
        self.input_name = info.name
        self.input_shape = info.shape  # (H, W, C)

    def infer(self, frame: np.ndarray) -> dict:
        with InferVStreams(
            self.network_group, self.input_params, self.output_params
        ) as pipeline:
            with self.network_group.activate(self.network_params):
                return pipeline.infer({self.input_name: frame[None, ...]})


class HailoFacePipeline:
    """Holds detector + embedder on a shared VDevice."""

    def __init__(self, detector_hef: Path, embedder_hef: Path):
        self.vdevice = VDevice()
        self.detector = _HailoModel(detector_hef, self.vdevice)
        self.embedder = _HailoModel(embedder_hef, self.vdevice)

    def close(self):
        self.vdevice.release()

    # ---- Detection ----
    def detect(
        self,
        bgr: np.ndarray,
        score_threshold: float,
        nms_iou: float,
    ) -> list[Detection]:
        in_h, in_w, _ = self.detector.input_shape
        resized, scale, pad = _letterbox(bgr, (in_w, in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)
        outputs = self.detector.infer(rgb)
        dets = _decode_scrfd(outputs, (in_w, in_h), score_threshold)
        dets = _nms(dets, nms_iou)
        # un-letterbox to original coords
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = (
                (x1 - pad[0]) / scale,
                (y1 - pad[1]) / scale,
                (x2 - pad[0]) / scale,
                (y2 - pad[1]) / scale,
            )
            d.landmarks = (d.landmarks - np.array(pad)) / scale
        return dets

    # ---- Embedding ----
    def embed(self, bgr_face_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_face_112, cv2.COLOR_BGR2RGB).astype(np.uint8)
        out = self.embedder.infer(rgb)
        # ArcFace HEFs typically expose a single output
        emb = next(iter(out.values())).reshape(-1).astype(np.float32)
        n = np.linalg.norm(emb) + 1e-9
        return emb / n


# ----------------- helpers -----------------


def _letterbox(img: np.ndarray, dst_wh: tuple[int, int]):
    src_h, src_w = img.shape[:2]
    dst_w, dst_h = dst_wh
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((dst_h, dst_w, 3), 114, dtype=np.uint8)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


# SCRFD strides and per-stride 2 anchors
_STRIDES = (8, 16, 32)
_ANCHORS_PER_LOC = 2


def _decode_scrfd(outputs: dict, input_wh: tuple[int, int], score_thr: float):
    """Decode SCRFD raw outputs into Detection list (in input image coords)."""
    in_w, in_h = input_wh
    # SCRFD heads come as 9 tensors: per stride [score, bbox, kps]
    # Sort by spatial size (large->small) so we map to strides 8,16,32
    arrs = [v[0] for v in outputs.values()]  # drop batch dim
    arrs.sort(key=lambda a: -a.shape[0] * a.shape[1])
    grouped = [arrs[i : i + 3] for i in range(0, len(arrs), 3)]

    dets: list[Detection] = []
    for stride, (score_t, bbox_t, kps_t) in zip(_STRIDES, grouped):
        h, w = score_t.shape[:2]
        scores = score_t.reshape(-1)
        bbox = bbox_t.reshape(-1, 4)
        kps = kps_t.reshape(-1, 10)

        ys, xs = np.mgrid[0:h, 0:w]
        anchor_centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        anchor_centers = np.repeat(
            anchor_centers.reshape(-1, 2), _ANCHORS_PER_LOC, axis=0
        )

        keep = scores >= score_thr
        if not np.any(keep):
            continue
        ac = anchor_centers[keep]
        bb = bbox[keep] * stride
        x1 = ac[:, 0] - bb[:, 0]
        y1 = ac[:, 1] - bb[:, 1]
        x2 = ac[:, 0] + bb[:, 2]
        y2 = ac[:, 1] + bb[:, 3]
        kp = kps[keep] * stride
        kp = kp.reshape(-1, 5, 2) + ac[:, None, :]
        for i in range(len(ac)):
            dets.append(
                Detection(
                    bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                    score=float(scores[keep][i]),
                    landmarks=kp[i].astype(np.float32),
                )
            )
    return dets


def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    if not dets:
        return []
    boxes = np.array([d.bbox for d in dets], dtype=np.float32)
    scores = np.array([d.score for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return [dets[i] for i in keep]


# Standard ArcFace 5-point reference landmarks for 112x112 alignment
_ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Affine-warp the face to 112x112 using 5-point similarity transform."""
    M, _ = cv2.estimateAffinePartial2D(landmarks, _ARCFACE_REF, method=cv2.LMEDS)
    if M is None:
        # Fallback: simple bbox crop+resize (lower accuracy)
        return cv2.resize(bgr, (112, 112))
    return cv2.warpAffine(bgr, M, (112, 112), borderValue=0)
