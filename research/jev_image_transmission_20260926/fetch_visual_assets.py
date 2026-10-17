"""Fetch pinned public Visual Jev inference sources and optional model assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "a392eaedfa41adae348f9767b55c0f54b3a13b57"
MODELS = {
    "qwen_base": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "visual_adapter": ("guanxuyu/visual-jev-4b-answer-sft", "7a3f1bb0d8f7bfbbd8514d968ab7de3518849c8d"),
}


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CognitiveSemCom-preflight"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "vendor/Visual-Jev"
    tree = get_json("https://api.github.com/repos/guanxuyu-sv/Visual-Jev/git/trees/" + COMMIT + "?recursive=1")["tree"]
    paths = [r["path"] for r in tree if r["type"] == "blob" and
             ((r["path"].startswith("code/vdm/") and r["path"].endswith(".py")) or
              r["path"] in ("code/examples/quickstart.py", "code/requirements.txt", "README.md", "REPRODUCE.md", "LICENSE"))]
    def fetch(path):
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with urllib.request.urlopen("https://raw.githubusercontent.com/guanxuyu-sv/Visual-Jev/" + COMMIT + "/" + path, timeout=30) as response:
                data = response.read()
            target.write_bytes(data)
        return {"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, paths))
    manifest = {"repository": "https://github.com/guanxuyu-sv/Visual-Jev", "commit": COMMIT,
                "files": files, "models": {k: {"repo_id": v[0], "revision": v[1]} for k, v in MODELS.items()}}
    (ROOT / "visual_assets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"source_files": len(files), "commit": COMMIT}), flush=True)
    if args.models:
        from huggingface_hub import snapshot_download
        for label, (repo, revision) in MODELS.items():
            print("DOWNLOADING " + label, flush=True)
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(ROOT / "models" / label),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                              ignore_patterns=["seed1/*", "seed2/*"], token=False, max_workers=2)
            print("READY " + label, flush=True)


if __name__ == "__main__":
    main()
