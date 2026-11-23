"""Drive cache helpers embedded into H3_Studio.ipynb (needs valid_model from model_cache.py)."""
import os
import time
from pathlib import Path


def file_ok(path, size=None):
    """Complete file: expected size when known, plus a full safetensors header/data check."""
    path = Path(path)
    try:
        if not path.is_file() or (size and path.stat().st_size != size):
            return False
    except OSError:
        return False
    return valid_model(path) if path.name.endswith(('.safetensors', '.safetensors.part')) else path.stat().st_size > 0


def copy_file(source, target, size=None):
    """Copy through a .part file and publish only after the copy checks out."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    total = source.stat().st_size
    copied, last = 0, time.monotonic()
    print(f'  複製 {source.name}（{total / 2**30:.1f} GiB）→ {target.parent}', flush=True)
    try:
        with source.open('rb') as src, partial.open('wb') as dst:
            while chunk := src.read(16 * 2**20):
                dst.write(chunk)
                copied += len(chunk)
                if time.monotonic() - last >= 5:
                    print(f'    {copied / 2**30:.1f} / {total / 2**30:.1f} GiB', flush=True)
                    last = time.monotonic()
        if not file_ok(partial, size or total):
            raise RuntimeError(f'複製後的檔案不完整：{target.name}')
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
