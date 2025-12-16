# utils/io_utils.py
from PIL import Image
import tempfile, os
from pathlib import Path

def save_image_temp(image: Image.Image, prefix: str = "aimh") -> str:
    tmpdir = "/tmp" if os.name != "nt" else os.getenv("TEMP", ".")
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(tmpdir, f"{prefix}_{abs(hash(image.tobytes()))%100000}.png")
    image.save(fpath)
    return fpath
