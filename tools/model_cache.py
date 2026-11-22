"""Helpers embedded into Colab notebooks for safe model reuse."""
import json
import os
import time
from pathlib import Path


def valid_model(path):
    """Check that a safetensors file has a complete header and data section."""
    try:
        path = Path(path)
        size = path.stat().st_size
        with path.open('rb') as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                return False
            header_size = int.from_bytes(prefix, 'little')
            if not 2 <= header_size <= min(size - 8, 100_000_000):
                return False
            header = json.loads(f.read(header_size))
        if not isinstance(header, dict):
            return False
        ranges = []
        for key, value in header.items():
            if key == '__metadata__':
                continue
            start, end = value['data_offsets']
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                return False
            ranges.append((start, end))
        if not ranges:
            return False
        position = 0
        for start, end in sorted(ranges):
            if start != position:
                return False
            position = end
        return 8 + header_size + position == size
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return False


def copy_model(source, target):
    """Copy to a temporary file, reporting progress and publishing only when complete."""
    source, target = Path(source), Path(target)
    partial = target.with_name(target.name + '.part')
    total = source.stat().st_size
    copied = 0
    last_update = time.monotonic()
    print(f'複製 {source.name}（{total / 2**30:.1f} GiB）', flush=True)
    try:
        with source.open('rb') as src, partial.open('wb') as dst:
            while True:
                chunk = src.read(8 * 2**20)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                if time.monotonic() - last_update >= 3:
                    print(f'  {copied / 2**30:.1f} / {total / 2**30:.1f} GiB', flush=True)
                    last_update = time.monotonic()
        if not valid_model(partial):
            raise RuntimeError(f'複製後的模型不完整：{target.name}')
        os.replace(partial, target)
        print('  複製完成', flush=True)
    finally:
        partial.unlink(missing_ok=True)


def ensure_model(repo, name, target, cached=None):
    """Reuse complete models; repair files left incomplete by interrupted copies."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(cached) if cached else None
    if not valid_model(target):
        if target.exists():
            print(f'伺服器模型不完整，重新取得：{target.name}', flush=True)
        if cached and valid_model(cached):
            print('從雲端硬碟還原', flush=True)
            copy_model(cached, target)
        else:
            if cached and cached.exists():
                print(f'雲端硬碟模型不完整，重新下載：{cached.name}', flush=True)
            got = Path(hf_hub_download(repo_id=repo, filename=name, local_dir='/content/hf'))
            if not valid_model(got):
                raise RuntimeError(f'下載的模型不完整：{name}')
            os.replace(got, target)
    if cached and not valid_model(cached):
        print('備份到雲端硬碟', flush=True)
        copy_model(target, cached)
