import os

def resolve_relative_path(path : str) -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def serialize_faces_analysis(detected_faces):
    for detected_face in detected_faces:
        detected_face.bbox = detected_face.bbox.tolist()
        detected_face.kps = detected_face.kps.tolist()
        detected_face.embedding = detected_face.embedding.tolist()
        detected_face.embedding_normed = detected_face.normed_embedding.tolist()
        detected_face.norm_embedding = float(detected_face.embedding_norm)
        detected_face.landmark_3d_68 = detected_face.landmark_3d_68.tolist()
        detected_face.pose = detected_face.pose.tolist()
        detected_face.landmark_2d_106 = detected_face.landmark_2d_106.tolist()
        detected_face.det_score = float(detected_face.det_score)
        detected_face.age = int(detected_face.age)
        detected_face.gender = int(detected_face.gender)
        detected_face = detected_face.__dict__
    
    return [detected_face.__dict__ for detected_face in detected_faces]


# onnx_providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
onnx_providers = ['CPUExecutionProvider']



# timer.py

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "{} - Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(self.name, elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()