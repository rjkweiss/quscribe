"""
PNG image payloads that every vision-language model client can send.
"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from quscribe.errors import InvalidInputError

# Vision APIs downsample large images anyway;
# shrinking first bounds token cost.
MAX_LONG_EDGE_PIXELS = 1568
PNG_MEDIA_TYPE = "image/png"
TRANSPARENT_MODES = frozenset({"RGBA", "LA", "PA"})


@dataclass(frozen=True)
class ImagePayload:
    """
    A normalized, PNG-encoded image ready to send to a vision-language model.

    Attributes:
        png_bytes: The PNG-encoded image data.
    """

    png_bytes: bytes

    @property
    def media_type(self) -> str:
        """
        Return the MIME type of the encoded image.

        Args:
            None.

        Returns:
            str: Always ``"image/png"``.
        """
        return PNG_MEDIA_TYPE

    @classmethod
    def from_pil_image(cls, image: Image.Image) -> "ImagePayload":
        """
        Normalize a Pillow image to an RGB PNG no larger than the size limit.

        Args:
            image: Any Pillow image - transparency is flattened onto white.

        Returns:
            ImagePayload: The normalized payload.
        """
        rgb_image = _flatten_onto_white(image)
        # thumbnail only ever shrinks and preserves the aspect ratio.
        rgb_image.thumbnail((MAX_LONG_EDGE_PIXELS, MAX_LONG_EDGE_PIXELS))
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="PNG")

        return cls(png_bytes=buffer.getvalue())

    @classmethod
    def from_bytes(cls, image_bytes: bytes) -> "ImagePayload":
        """
        Decode raw image bytes in a Pillow-supported format and normalize them.

        Args:
            image_bytes: Encoded image data (PNG, JPEG, WebP, ...).

        Returns:
            ImagePayload: The normalized payload.

        Raises:
            InvalidInputError: If the bytes are not a readable image.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return cls.from_pil_image(image)
        except OSError as error:  # Pillow's UnidentifiedImageError is OSError.
            raise InvalidInputError(
                f"The data is not a readable image: {error}"
            ) from error

    @classmethod
    def from_file(cls, image_path: str | Path) -> "ImagePayload":
        """
        Read and normalize an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            ImagePayload: The normalized payload.

        Raises:
            InvalidInputError: If the file cannot be read or decoded.
        """
        try:
            image_bytes = Path(image_path).read_bytes()
        except OSError as error:
            raise InvalidInputError(
                f"Cannot read image file {image_path}: {error}"
            ) from error

        return cls.from_bytes(image_bytes)

    def to_base64(self) -> str:
        """
        Encode the PNG data as base64 text, as model APIs expect.

        Args:
            None.

        Returns:
            str: The base64-encoded PNG data.
        """
        return base64.b64encode(self.png_bytes).decode("ascii")

    def to_pil_image(self) -> Image.Image:
        """
        Decode the payload into a Pillow image for display in the web app.

        Args:
            None.

        Returns:
            Image.Image: The decoded image.
        """
        image = Image.open(io.BytesIO(self.png_bytes))
        image.load()  # Decode so the image no longer depends on the buffer.

        return image


def _flatten_onto_white(image: Image.Image) -> Image.Image:
    """
    Convert an image to RGB, compositing any transparency onto a
    white background.

    Args:
        image: Any Pillow image.

    Returns:
        Image.Image: A new RGB image.
    """
    has_transparency = image.mode in TRANSPARENT_MODES or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_transparency:
        return image.convert("RGB")

    rgba_image = image.convert("RGBA")
    background = Image.new("RGB", rgba_image.size, "white")
    background.paste(rgba_image, mask=rgba_image.getchannel("A"))

    return background
