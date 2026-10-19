"""Download all AI weights once (used on first run).

Uses Windows curl.exe under the hood for large files because Python httpx on Windows
can stall after 10 MB when streaming from HF CDN.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from huggingface_hub import file_download, hf_hub_download  # noqa: E402

from mesh_core.config import TEXT2IMG_MODEL_ID, TRIPOSR_MODEL_ID  # noqa: E402

HUNYUAN_MINI_ID = "tencent/Hunyuan3D-2mini"


def _install_curl_http_get() -> None:
    """Replace huggingface_hub's http_get with curl.exe for rock-solid downloads."""
    orig_http_get = file_download.http_get

    def curl_http_get(url, temp_file, *, resume_size=0, headers=None, expected_size=None, **_kw):
        if expected_size is not None and expected_size < 1_000_000:
            return orig_http_get(
                url, temp_file, resume_size=resume_size, headers=headers,
                expected_size=expected_size, **_kw
            )
        temp_file.flush()
        path = Path(temp_file.name)
        temp_file.close()
        cmd = [
            "curl.exe", "-L", "--fail", "--retry", "5", "--retry-delay", "2",
            "-C", "-", "-o", str(path),
        ]
        if headers:
            for k, v in headers.items():
                if k.lower() != "range":
                    cmd += ["-H", f"{k}: {v}"]
        cmd.append(url)
        mb = f"{expected_size / (1024*1024):.1f} MB" if expected_size else "unknown size"
        print(f"     [curl.exe] Tải {mb} -> {path.name} ...", flush=True)
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"curl.exe failed with exit code {res.returncode}")

    file_download.http_get = curl_http_get


_install_curl_http_get()

SD_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

TRIPOSR_FILES = [
    "config.yaml",
    "model.ckpt",
]

HUNYUAN_FILES = [
    "hunyuan3d-dit-v2-mini-turbo/config.yaml",
    "hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors",
    "hunyuan3d-vae-v2-mini-turbo/config.yaml",
    "hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors",
]


def main() -> None:
    print(f"[1/3] Text-to-image: {TEXT2IMG_MODEL_ID}", flush=True)
    for f in SD_FILES:
        hf_hub_download(repo_id=TEXT2IMG_MODEL_ID, filename=f)

    print(f"[2/3] Image-to-3D (Nhanh): {TRIPOSR_MODEL_ID}", flush=True)
    for f in TRIPOSR_FILES:
        hf_hub_download(repo_id=TRIPOSR_MODEL_ID, filename=f)

    print(f"[3/3] Image-to-3D (Chất lượng cao): {HUNYUAN_MINI_ID}", flush=True)
    for f in HUNYUAN_FILES:
        print(f"  -> Kiểm tra / tải {f} ...", flush=True)
        hf_hub_download(repo_id=HUNYUAN_MINI_ID, filename=f)

    print("Hoàn tất tải toàn bộ model!", flush=True)


if __name__ == "__main__":
    main()
