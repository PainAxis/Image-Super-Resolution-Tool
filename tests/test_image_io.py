"""Round-trip fidelity and failure-safety tests for image I/O."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from sr_tool.utils.image_io import (
    ImageIOError,
    ImageMetadata,
    load_image,
    load_image_document,
    premultiply_rgb,
    resize_alpha,
    save_image,
    save_image_document,
    unpremultiply_rgb,
)


def test_compatibility_load_returns_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8)).save(path)
    result = load_image(path)
    assert result.shape == (3, 4, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 128 / 255, atol=1e-7)


def test_alpha_is_preserved_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = [[0, 64, 128], [192, 255, 32]]
    Image.fromarray(rgba).save(path)
    document = load_image_document(path)
    assert document.alpha is not None
    np.testing.assert_array_equal(
        np.rint(document.alpha * 255).astype(np.uint8), rgba[..., 3]
    )
    output = tmp_path / "alpha-output.png"
    save_image_document(document, output)
    np.testing.assert_array_equal(np.asarray(Image.open(output)), rgba)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "oriented.png"
    pixels = np.zeros((3, 2, 3), dtype=np.uint8)
    pixels[0, 0] = [255, 0, 0]
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(path, exif=exif)
    document = load_image_document(path)
    assert document.rgb.shape == (2, 3, 3)
    np.testing.assert_allclose(document.rgb[0, 2], [1.0, 0.0, 0.0])


def test_sixteen_bit_grayscale_is_lossless(tmp_path: Path) -> None:
    values = np.array([[0, 1, 255, 256, 32768, 65535]], dtype=np.uint16)
    source = tmp_path / "16-bit.png"
    Image.fromarray(values).save(source)
    document = load_image_document(source)
    assert document.metadata.bit_depth == 16
    np.testing.assert_allclose(document.rgb[0, :, 0], values[0] / 65535.0, atol=1e-8)
    output = tmp_path / "16-bit-output.png"
    save_image_document(document, output)
    saved = np.asarray(Image.open(output))
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(saved, values)


def test_sixteen_bit_color_is_rejected_before_pillow_reduces_it(
    tmp_path: Path,
) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)
    scanline = b"\x00\xff\xff\x00\x00\x00\x00"
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanline))
        + chunk(b"IEND", b"")
    )
    path = tmp_path / "rgb16.png"
    path.write_bytes(encoded)
    with pytest.raises(ImageIOError, match="16-bit color PNG"):
        load_image_document(path)


def test_icc_profile_is_converted_and_preserved(tmp_path: Path) -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "profiled.png"
    Image.fromarray(np.full((2, 2, 3), 100, dtype=np.uint8)).save(
        source, icc_profile=profile
    )
    document = load_image_document(source)
    assert document.metadata.icc_profile
    output = tmp_path / "profiled-output.png"
    save_image_document(document, output)
    with Image.open(output) as saved:
        assert saved.info.get("icc_profile")


def test_alpha_resize_contract() -> None:
    alpha = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    result = resize_alpha(alpha, (7, 9))
    assert result.shape == (7, 9)
    assert result.dtype == np.float32
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_premultiplied_alpha_round_trip_and_zero_handling() -> None:
    rgb = np.array([[[1.0, 0.5, 0.25], [1.0, 0.0, 0.0]]], dtype=np.float32)
    alpha = np.array([[0.5, 0.0]], dtype=np.float32)
    premultiplied = premultiply_rgb(rgb, alpha)
    restored = unpremultiply_rgb(premultiplied, alpha)
    np.testing.assert_allclose(restored[0, 0], rgb[0, 0])
    np.testing.assert_array_equal(restored[0, 1], 0.0)


def test_incompatible_formats_fail_instead_of_dropping_data(tmp_path: Path) -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    alpha = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ImageIOError, match="transparency"):
        save_image(rgb, tmp_path / "alpha.jpg", alpha=alpha)
    with pytest.raises(ImageIOError, match="16-bit"):
        save_image(
            rgb,
            tmp_path / "deep.jpg",
            metadata=ImageMetadata(bit_depth=16, grayscale=True),
        )


def test_failed_save_does_not_replace_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.png"
    destination.write_bytes(b"previous")

    def fail_save(self: Image.Image, *args: object, **kwargs: object) -> None:
        raise OSError("simulated encoder failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        save_image(np.zeros((2, 2, 3), dtype=np.float32), destination)
    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]
