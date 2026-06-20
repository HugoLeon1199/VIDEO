"""Exceptions for the image generation backend."""


class ImageGenerationError(Exception):
    """Base error for all image generation failures."""


class ValidationError(ImageGenerationError):
    """Invalid request parameters."""


class BackendUnavailableError(ImageGenerationError):
    """Backend endpoint unreachable or misconfigured."""


class JobTimeoutError(ImageGenerationError):
    """Job did not complete within the allowed time."""


class JobFailedError(ImageGenerationError):
    """RunPod returned a failed/cancelled status."""


class ChecksumMismatchError(ImageGenerationError):
    """Downloaded image SHA-256 does not match returned checksum."""


class PayloadTooLargeError(ImageGenerationError):
    """base64 payload exceeds safe size limit — switch to volume mode."""
