"""Metadata-aware, bounded and atomic image input/output."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps, PngImagePlugin

from sr_tool.fsr.common import validate_rgb_image
from sr_tool.utils.resources import ensure_input_size

_TEXT_METADATA_LIMIT = 1024 * 1024
_ALPHA_FORMATS = {".png", ".tif", ".tiff"}
_SIXTEEN_BIT_FORMATS = {".png", ".tif", ".tiff"}
_ICC_FORMATS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_EXIF_FORMATS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class ImageIOError(ValueError):
    """Raised for a supported file with an unsafe or unsupported conversion."""


@dataclass(frozen=True)
class ImageMetadata:
    """Safe metadata that can be carried to an upscaled output."""

    exif: bytes | None = None
    icc_profile: bytes | None = None
    dpi: tuple[float, float] | None = None
    source_format: str | None = None
    bit_depth: int = 8
    grayscale: bool = False
    png_text: tuple[tuple[str, str], ...] = ()
    comment: bytes | str | None = None


@dataclass(frozen=True)
class ImageDocument:
    """Decoded RGB pixels plus optional straight alpha and metadata."""

    rgb: np.ndarray
    alpha: np.ndarray | None
    metadata: ImageMetadata

    def with_pixels(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray | None = None,
    ) -> ImageDocument:
        return ImageDocument(rgb=rgb, alpha=alpha, metadata=self.metadata)


def _integer_tag_values(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _source_bit_depth(
    source_path: Path,
    opened: Image.Image,
) -> int:
    """Return a supported source depth or reject lossy Pillow coercions."""
    if opened.format == "PNG":
        try:
            with source_path.open("rb") as source:
                header = source.read(26)
        except OSError as exc:
            raise ImageIOError(f"Could not inspect PNG header: {exc}") from exc
        png_signature = b"\x89PNG\r\n\x1a\n"
        if not (
            len(header) >= 26
            and header[:8] == png_signature
            and header[12:16] == b"IHDR"
        ):
            raise ImageIOError("Could not inspect PNG precision")
        bit_depth = int(header[24])
        color_type = int(header[25])
        if bit_depth == 16:
            if color_type != 0:
                raise ImageIOError(
                    "16-bit color PNG is not supported without precision loss; "
                    "convert to 16-bit grayscale or an 8-bit color image first"
                )
            return 16
        return 8

    if opened.format == "TIFF" and hasattr(opened, "tag_v2"):
        bits = _integer_tag_values(opened.tag_v2.get(258))
        sample_formats = _integer_tag_values(opened.tag_v2.get(339)) or (1,)
        if not bits and (opened.mode == "I" or opened.mode.startswith("I;16")):
            raise ImageIOError(
                "Could not determine TIFF integer precision without risking data loss"
            )
        if not bits and opened.mode == "F":
            raise ImageIOError(
                "Floating-point TIFF is not supported without precision loss"
            )
        if any(sample_format == 3 for sample_format in sample_formats):
            raise ImageIOError(
                "Floating-point TIFF is not supported without precision loss"
            )
        if any(sample_format == 2 for sample_format in sample_formats):
            precision = max(bits, default=0)
            raise ImageIOError(
                f"Signed {precision}-bit integer TIFF is not supported without "
                "precision loss"
            )
        if any(sample_format != 1 for sample_format in sample_formats):
            raise ImageIOError("Unsupported TIFF sample format")
        if any(bit_depth > 8 for bit_depth in bits):
            if bits == (16,):
                return 16
            precision = max(bits)
            kind = "color" if len(bits) > 1 else "grayscale"
            raise ImageIOError(
                f"{precision}-bit {kind} TIFF is not supported without precision "
                "loss; convert to 16-bit grayscale or an 8-bit image first"
            )
        return 8

    if opened.mode == "F":
        raise ImageIOError(
            "Floating-point image data is not supported without precision loss"
        )
    if opened.mode == "I" or opened.mode.startswith("I;16"):
        raise ImageIOError(
            "This integer image precision is not supported without precision loss"
        )
    return 8


def _bounded_png_text(img: Image.Image) -> tuple[tuple[str, str], ...]:
    raw = getattr(img, "text", {})
    result: list[tuple[str, str]] = []
    used = 0
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            item_size = len(key.encode("utf-8")) + len(value.encode("utf-8"))
            if used + item_size > _TEXT_METADATA_LIMIT:
                break
            result.append((key, value))
            used += item_size
    return tuple(result)


def _normalized_dpi(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        dpi = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(component) and component > 0.0 for component in dpi):
        return None
    return dpi


def _convert_profile(
    source_image: Image.Image,
    icc_profile: bytes | None,
) -> tuple[Image.Image, bytes | None]:
    """Convert an embedded profile to sRGB instead of merely relabeling data."""
    if not icc_profile:
        return source_image.convert("RGB"), None
    try:
        source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
        destination_profile = ImageCms.createProfile("sRGB")
        try:
            converted = ImageCms.profileToProfile(
                source_image,
                source_profile,
                destination_profile,
                outputMode="RGB",
            )
        except (OSError, ValueError, ImageCms.PyCMSError):
            if source_image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                raise
            converted = ImageCms.profileToProfile(
                source_image.convert("RGB"),
                source_profile,
                destination_profile,
                outputMode="RGB",
            )
        if converted is None:
            raise ImageIOError("ICC conversion returned no image")
        destination_bytes = ImageCms.ImageCmsProfile(destination_profile).tobytes()
        return converted, destination_bytes
    except (OSError, ValueError, ImageCms.PyCMSError) as exc:
        raise ImageIOError(f"Invalid or unsupported ICC profile: {exc}") from exc


def load_image_document(path: str | os.PathLike[str]) -> ImageDocument:
    """Decode an image with orientation, alpha, profile and bit-depth context."""
    source_path = Path(path)
    try:
        with Image.open(source_path) as opened:
            ensure_input_size(opened.height, opened.width)
            bit_depth = _source_bit_depth(source_path, opened)
            source_format = opened.format
            source_info = dict(opened.info)
            oriented = ImageOps.exif_transpose(opened)
            ensure_input_size(oriented.height, oriented.width)
            mode = oriented.mode
            grayscale = mode in {"1", "L", "LA", "I", "F"} or mode.startswith("I;16")

            alpha: np.ndarray | None = None
            has_alpha = "A" in mode or (mode == "P" and "transparency" in source_info)
            if has_alpha:
                alpha_image = oriented.convert("RGBA").getchannel("A")
                alpha = np.asarray(alpha_image, dtype=np.float32) / np.float32(255.0)

            if bit_depth == 16:
                raw = np.asarray(oriented)
                if raw.size == 0 or int(np.min(raw)) < 0 or int(np.max(raw)) > 65535:
                    raise ImageIOError(
                        "Only unsigned 16-bit integer images are supported"
                    )
                gray = raw.astype(np.float32) / np.float32(65535.0)
                rgb = np.repeat(gray[..., None], 3, axis=2)
                output_profile = source_info.get("icc_profile")
            else:
                rgb_image, output_profile = _convert_profile(
                    oriented, source_info.get("icc_profile")
                )
                rgb = np.asarray(rgb_image, dtype=np.float32) / np.float32(255.0)

            exif = oriented.getexif()
            exif_bytes = exif.tobytes() if len(exif) else None
            comment = source_info.get("comment")
            if (
                isinstance(comment, (bytes, str))
                and len(comment) > _TEXT_METADATA_LIMIT
            ):
                comment = None
            metadata = ImageMetadata(
                exif=exif_bytes,
                icc_profile=output_profile,
                dpi=_normalized_dpi(source_info.get("dpi")),
                source_format=source_format,
                bit_depth=bit_depth,
                grayscale=grayscale,
                png_text=_bounded_png_text(oriented),
                comment=comment if isinstance(comment, (bytes, str)) else None,
            )
    except (Image.DecompressionBombError, OSError) as exc:
        raise ImageIOError(f"Could not decode image: {exc}") from exc

    return ImageDocument(
        rgb=np.asarray(rgb, dtype=np.float32),
        alpha=None if alpha is None else np.asarray(alpha, dtype=np.float32),
        metadata=metadata,
    )


def load_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Compatibility API returning only normalized float32 RGB pixels."""
    return load_image_document(path).rgb


def resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize straight alpha to ``(height, width)`` with a Lanczos filter."""
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim != 2 or not np.isfinite(alpha).all():
        raise ValueError("alpha must be a finite 2-D array")
    height, width = size
    if height < 1 or width < 1:
        raise ValueError("alpha output dimensions must be positive")
    image = Image.fromarray(np.clip(alpha, 0.0, 1.0))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def premultiply_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Premultiply RGB before spatial filtering to prevent transparent fringes."""
    source = validate_rgb_image(rgb)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.shape != source.shape[:2] or not np.isfinite(alpha).all():
        raise ValueError("alpha shape must match RGB height and width")
    return source * np.clip(alpha, 0.0, 1.0)[..., None]


def unpremultiply_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Convert filtered premultiplied RGB back to straight alpha safely."""
    source = validate_rgb_image(rgb)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.shape != source.shape[:2] or not np.isfinite(alpha).all():
        raise ValueError("alpha shape must match RGB height and width")
    result = np.zeros_like(source)
    safe_alpha = np.clip(alpha, 0.0, 1.0)
    np.divide(
        source,
        safe_alpha[..., None],
        out=result,
        where=safe_alpha[..., None] > np.float32(1.0 / 65535.0),
    )
    return np.clip(result, 0.0, 1.0)


def _updated_exif(exif_bytes: bytes | None, width: int, height: int) -> bytes | None:
    if not exif_bytes:
        return None
    try:
        exif = Image.Exif()
        exif.load(exif_bytes)
        exif[274] = 1  # Orientation was normalized during decode.
        exif[256] = width
        exif[257] = height
        exif[40962] = width
        exif[40963] = height
        return exif.tobytes()
    except (OSError, ValueError, TypeError):
        return None


def _encode_image(
    arr: np.ndarray,
    extension: str,
    alpha: np.ndarray | None,
    metadata: ImageMetadata,
) -> tuple[Image.Image, dict[str, Any]]:
    source = validate_rgb_image(arr)
    height, width = source.shape[:2]
    extension = extension.lower()
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=np.float32)
        if alpha.shape != (height, width) or not np.isfinite(alpha).all():
            raise ValueError("alpha shape must match RGB height and width")
        if extension not in _ALPHA_FORMATS:
            raise ImageIOError(
                "This output format cannot preserve transparency; use PNG or TIFF"
            )

    if metadata.bit_depth not in {8, 16}:
        raise ImageIOError("Only 8-bit and 16-bit image metadata is supported")
    if metadata.bit_depth == 16:
        if extension not in _SIXTEEN_BIT_FORMATS:
            raise ImageIOError(
                "This output format cannot preserve 16-bit data; use PNG or TIFF"
            )
        if alpha is not None:
            raise ImageIOError(
                "16-bit images with alpha are not supported by this encoder"
            )
        if not metadata.grayscale:
            raise ImageIOError(
                "16-bit color output is not supported; choose an 8-bit export"
            )
        gray = np.mean(source, axis=2)
        image = Image.fromarray(np.rint(gray * 65535.0).astype(np.uint16))
    else:
        rgb8 = np.rint(np.clip(source, 0.0, 1.0) * 255.0).astype(np.uint8)
        if alpha is None:
            image = Image.fromarray(rgb8)
        else:
            alpha8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
            image = Image.fromarray(np.dstack((rgb8, alpha8)))

    options: dict[str, Any] = {}
    if metadata.icc_profile and extension in _ICC_FORMATS:
        options["icc_profile"] = metadata.icc_profile
    exif = _updated_exif(metadata.exif, width, height)
    if exif and extension in _EXIF_FORMATS:
        options["exif"] = exif
    if metadata.dpi:
        options["dpi"] = metadata.dpi
    if extension == ".png" and metadata.png_text:
        png_info = PngImagePlugin.PngInfo()
        for key, value in metadata.png_text:
            png_info.add_text(key, value)
        options["pnginfo"] = png_info
    if extension in {".jpg", ".jpeg"}:
        options.update({"quality": 95, "subsampling": 0})
        if metadata.comment is not None:
            options["comment"] = metadata.comment
    return image, options


def _sync_parent_directory(directory: Path) -> None:
    """Best-effort directory sync after replace on supporting POSIX filesystems."""
    if os.name == "nt":
        return
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError:
        # Directory fsync is unavailable on some macOS, network and FUSE
        # filesystems. The file itself was synced before the atomic replace,
        # so lack of directory durability must not turn a successful save into
        # a false failure after the destination has already changed.
        pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def save_image(
    arr: np.ndarray,
    path: str | os.PathLike[str],
    *,
    alpha: np.ndarray | None = None,
    metadata: ImageMetadata | None = None,
) -> None:
    """Atomically save an image, retaining safe metadata when supplied."""
    destination = Path(path)
    if not destination.suffix:
        raise ImageIOError("Output filename must have an extension")
    if not destination.parent.exists():
        raise ImageIOError(f"Output directory does not exist: {destination.parent}")
    metadata = metadata or ImageMetadata()
    image, options = _encode_image(arr, destination.suffix, alpha, metadata)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, **options)
        # Windows implements fsync via the writable-handle-only _commit().
        # Reopen read/write so the durability fence is portable.
        with temporary.open("r+b") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, destination)
        _sync_parent_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_image_document(
    document: ImageDocument,
    path: str | os.PathLike[str],
) -> None:
    save_image(
        document.rgb,
        path,
        alpha=document.alpha,
        metadata=document.metadata,
    )


def array_to_photoimage(arr: np.ndarray) -> object:
    """Convert normalized RGB data to a Tk ``PhotoImage``."""
    source = validate_rgb_image(arr)
    uint8_arr = np.rint(source * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_arr)
    from PIL import ImageTk

    return ImageTk.PhotoImage(pil_img)
