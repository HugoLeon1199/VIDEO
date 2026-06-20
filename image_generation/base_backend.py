"""Abstract base class for image generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from image_generation.schemas import SceneRequest, SceneResult


class BaseImageBackend(ABC):
    @abstractmethod
    def generate(self, request: SceneRequest) -> SceneResult:
        """Generate images for one scene. Must be thread-safe."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and configured."""
