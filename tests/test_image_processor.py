import asyncio
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.image_processor import ImageProcessor, InpaintRegion


class ImageProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_mask_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            destination = Path(directory) / "result.png"
            Image.new("RGB", (32, 32), "red").save(source)
            processor = ImageProcessor(asyncio.Semaphore(1))
            await processor.process(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())

    async def test_inpainting_keeps_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            destination = Path(directory) / "result.png"
            image = Image.new("RGB", (48, 40), "white")
            for x in range(18, 30):
                for y in range(15, 25):
                    image.putpixel((x, y), (0, 0, 0))
            image.save(source)
            processor = ImageProcessor(asyncio.Semaphore(1))
            await processor.process(
                source,
                destination,
                regions=[InpaintRegion(x=18, y=15, width=12, height=10)],
            )
            with Image.open(destination) as result:
                self.assertEqual(result.size, (48, 40))


if __name__ == "__main__":
    unittest.main()
