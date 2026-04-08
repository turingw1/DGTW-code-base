from __future__ import annotations

from .base import NullTeacher
from .dummy import DummyTeacher
from .diffusers_ddpm import DiffusersDDPMTeacher


def build_teacher(config: dict):
    teacher_cfg = config.get("teacher", {})
    teacher_type = str(teacher_cfg.get("type", "none"))
    if teacher_type == "none":
        return NullTeacher()
    if teacher_type == "sampler":
        backend = str(teacher_cfg.get("backend", "diffusers_ddpm"))
        if backend == "dummy":
            return DummyTeacher(config)
        if backend != "diffusers_ddpm":
            raise ValueError(f"Unsupported sampler teacher backend: {backend}")
        return DiffusersDDPMTeacher(config)
    raise ValueError(f"Unsupported teacher.type: {teacher_type}")
