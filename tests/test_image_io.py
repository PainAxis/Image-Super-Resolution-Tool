"""Round-trip fidelity and failure-safety tests for image I/O."""

import stat
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from sr_tool.utils import image_io
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


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        (
            "signed-32-bit",
            np.array([[0, 1, 65535]], dtype=np.int32),
            "32-bit integer TIFF",
        ),
        (
            "floating-point",
            np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
            "Floating-point TIFF",
        ),
    ],
)
def test_unsupported_tiff_precision_is_rejected_instead_of_requantized(
    tmp_path: Path,
    name: str,
    values: np.ndarray,
    message: str,
) -> None:
    source = tmp_path / f"{name}.tiff"
    Image.fromarray(values).save(source)
    with pytest.raises(ImageIOError, match=message):
        load_image_document(source)


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


def test_non_rgb_icc_profile_is_converted_from_its_native_mode(
    tmp_path: Path,
) -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    source = tmp_path / "profiled-lab.tiff"
    Image.new("LAB", (2, 2), (128, 140, 120)).save(source, icc_profile=profile)

    document = load_image_document(source)

    assert document.rgb.shape == (2, 2, 3)
    assert document.rgb.dtype == np.float32
    assert document.metadata.icc_profile
    np.testing.assert_allclose(
        document.rgb[0, 0],
        np.array([135, 113, 133], dtype=np.float32) / 255.0,
        atol=1.0 / 255.0,
    )


def test_palette_icc_profile_uses_rgb_fallback_and_preserves_alpha(
    tmp_path: Path,
) -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    palette = Image.new("P", (2, 1))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 6))
    palette.putdata([0, 1])
    source = tmp_path / "profiled-palette.png"
    palette.save(source, icc_profile=profile, transparency=0)

    document = load_image_document(source)

    assert document.alpha is not None
    np.testing.assert_array_equal(document.alpha, [[0.0, 1.0]])
    np.testing.assert_allclose(document.rgb[0, 0], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(document.rgb[0, 1], [0.0, 1.0, 0.0], atol=1e-7)
    assert document.metadata.icc_profile


@pytest.mark.parametrize(
    ("mode", "pixels", "transparency", "expected_alpha"),
    [
        (
            "L",
            np.array([[0, 128, 255]], dtype=np.uint8),
            128,
            np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        ),
        (
            "RGB",
            np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
            (0, 255, 0),
            np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        ),
    ],
)
def test_png_transparency_keys_are_preserved(
    tmp_path: Path,
    mode: str,
    pixels: np.ndarray,
    transparency: int | tuple[int, int, int],
    expected_alpha: np.ndarray,
) -> None:
    source = tmp_path / f"transparency-{mode}.png"
    Image.fromarray(pixels).save(source, transparency=transparency)

    document = load_image_document(source)

    assert document.alpha is not None
    np.testing.assert_array_equal(document.alpha, expected_alpha)
    output = tmp_path / f"transparency-{mode}-output.png"
    save_image_document(document, output)
    with Image.open(output) as saved:
        saved_alpha = np.asarray(saved.convert("RGBA"), dtype=np.uint8)[..., 3]
    np.testing.assert_array_equal(saved_alpha, expected_alpha * 255)


def test_invalid_icc_profile_is_reported_as_an_image_error(tmp_path: Path) -> None:
    source = tmp_path / "invalid-profile.png"
    Image.new("RGB", (2, 2)).save(source, icc_profile=b"not-an-icc-profile")
    with pytest.raises(ImageIOError, match="Invalid or unsupported ICC profile"):
        load_image_document(source)


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
    with pytest.raises(ImageIOError, match="8-bit and 16-bit"):
        save_image(
            rgb,
            tmp_path / "unsupported-depth.png",
            metadata=ImageMetadata(bit_depth=32, grayscale=True),
        )


def test_bmp_omits_unsupported_color_profile_metadata(tmp_path: Path) -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    destination = tmp_path / "profiled.bmp"
    save_image(
        np.full((2, 2, 3), 0.5, dtype=np.float32),
        destination,
        metadata=ImageMetadata(icc_profile=profile, dpi=(96.0, 96.0)),
    )
    with Image.open(destination) as saved:
        assert "icc_profile" not in saved.info
        assert saved.info["dpi"] == pytest.approx((96.0, 96.0), rel=0.01)


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


def test_file_fsync_is_writable_and_directory_fsync_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = image_io.os.fsync

    def reject_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(image_io.os.fstat(descriptor).st_mode):
            raise OSError("directory sync is unsupported")
        # A zero-byte write leaves the image unchanged but fails with EBADF
        # when the descriptor was opened read-only, as the old implementation was.
        assert image_io.os.write(descriptor, b"") == 0
        real_fsync(descriptor)

    monkeypatch.setattr(image_io.os, "fsync", reject_directory_fsync)
    destination = tmp_path / "portable.png"
    save_image(np.full((2, 3, 3), 0.25, dtype=np.float32), destination)
    assert destination.exists()
    np.testing.assert_allclose(
        np.asarray(Image.open(destination), dtype=np.float32) / 255.0,
        0.25,
        atol=1.0 / 255.0,
    )
