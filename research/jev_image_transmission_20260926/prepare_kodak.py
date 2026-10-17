"""Acquire public development images; no original file is overwritten."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def prepare(index):
    name = f"kodim{index:02d}.png"
    url = "https://r0k.us/graphics/kodak/kodak/" + name
    raw_path = ROOT / "datasets/kodak/raw" / name
    prepared_path = ROOT / "datasets/kodak/128" / name
    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        raw_path.write_bytes(raw)
    with Image.open(io.BytesIO(raw)) as image:
        size = image.size
        assert size in ((768, 512), (512, 768))
        prepared = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        prepared.save(buffer, format="PNG")
        data = buffer.getvalue()
    if prepared_path.exists():
        assert prepared_path.read_bytes() == data
    else:
        prepared_path.write_bytes(data)
    return {"image_id": name, "url": url, "original_size": size,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "prepared_sha256": hashlib.sha256(data).hexdigest(),
            "role": "p1_development" if index <= 16 else "reserved"}


def main():
    for directory in ("raw", "128"):
        (ROOT / "datasets/kodak" / directory).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(prepare, range(1, 25)))
    manifest = {"source": "https://r0k.us/graphics/kodak/",
                "purpose": "Local development only, not formal held-out evaluation",
                "preprocessing": "EXIF transpose -> RGB -> bicubic 128x128 PNG", "images": entries}
    path = ROOT / "kodak_manifest.json"
    text = json.dumps(manifest, indent=2)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text
    else:
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"prepared": len(entries), "manifest": str(path)}), flush=True)


if __name__ == "__main__":
    main()
