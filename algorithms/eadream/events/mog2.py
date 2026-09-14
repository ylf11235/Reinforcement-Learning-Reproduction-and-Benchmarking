"""OpenCV MOG2 event masks and deterministic fixture generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class MOG2Config:
    """Explicit MOG2 defaults used by the pinned EADream event protocol."""

    history: int = 500
    var_threshold: float = 16.0
    detect_shadows: bool = True
    learning_rate: float = -1.0
    closing_kernel: int = 3
    closing_shape: Literal["ellipse"] = "ellipse"


def _validate_image(image: np.ndarray) -> np.ndarray:
    if type(image) is not np.ndarray:
        raise TypeError("image must be a plain numpy.ndarray")
    if image.shape != (64, 64, 3):
        raise ValueError("image must have shape (64, 64, 3)")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    return np.ascontiguousarray(image)


class MOG2EventExtractor:
    """Maintain one OpenCV MOG2 background model for one environment."""

    def __init__(self, config: MOG2Config) -> None:
        self.config = config
        self._subtractor: cv2.BackgroundSubtractor | None = None
        self._kernel: np.ndarray | None = None

    def reset(self, image: np.ndarray) -> np.ndarray:
        """Discard episode state and learn the post-reset image as background."""
        validated = _validate_image(image)
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )
        self._subtractor.apply(validated, learningRate=self.config.learning_rate)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.closing_kernel, self.config.closing_kernel),
        )
        return np.zeros(validated.shape[:2], dtype=np.uint8)

    def step(self, image: np.ndarray) -> np.ndarray:
        """Return the next closed foreground/shadow event mask."""
        if self._subtractor is None or self._kernel is None:
            raise RuntimeError("reset must be called before step")
        raw = self._subtractor.apply(
            _validate_image(image), learningRate=self.config.learning_rate
        )
        return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, self._kernel)


def _fixture_frames() -> np.ndarray:
    """Build the documented twelve-frame moving-square RGB trace."""
    frames = np.empty((12, 64, 64, 3), dtype=np.uint8)
    background = np.empty((64, 64, 3), dtype=np.uint8)
    background[..., 0] = 24
    background[..., 1] = 56
    background[..., 2] = 88
    background[40:48, 4:20] = (40, 72, 104)
    for index in range(len(frames)):
        frames[index] = background
        top = 4 + index * 3
        left = 4 + index * 4
        frames[index, top : top + 8, left : left + 8] = (255, 255, 255)
    return frames


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_fixture(directory: Path) -> dict[str, object]:
    """Write a reproducible 12-frame MOG2 golden trace and provenance manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = _fixture_frames()
    extractor = MOG2EventExtractor(MOG2Config())
    events = np.empty((len(frames), 64, 64), dtype=np.uint8)
    events[0] = extractor.reset(frames[0])
    for index, frame in enumerate(frames[1:], start=1):
        events[index] = extractor.step(frame)

    np.savez_compressed(directory / "mog2_input.npz", frames=frames)
    np.savez_compressed(directory / "mog2_expected.npz", events=events)
    manifest: dict[str, object] = {
        "fixture": {
            "description": "Twelve 64x64 RGB frames with one moving 8x8 white square over a fixed background.",
            "frame_count": len(frames),
            "frame_shape": list(frames.shape[1:]),
            "generator": "eadream.events.mog2._fixture_frames",
        },
        "arrays": {
            "input.frames.sha256": _sha256(frames),
            "expected.events.sha256": _sha256(events),
        },
        "dependencies": {
            "numpy": np.__version__,
            "opencv-python-headless": version("opencv-python-headless"),
            "cv2": cv2.__version__,
        },
        "mog2_config": asdict(MOG2Config()),
    }
    (directory / "mog2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixture", type=Path, metavar="DIRECTORY")
    arguments = parser.parse_args(argv)
    if arguments.write_fixture is None:
        parser.error("--write-fixture is required")
    write_fixture(arguments.write_fixture)


if __name__ == "__main__":
    main()
