"""One flat annotations directory shared by desktop and web review."""
import json
import os
from pathlib import Path
import tempfile

from launcher.training_material import clean_reviews


def review_path(batch_dir: Path) -> Path:
    batch = Path(batch_dir).resolve()
    root = batch.parent.parent / 'annotations'
    path = root / (batch.name + '.json')
    if path.resolve().parent != root.resolve():
        raise ValueError('标注保存路径无效')
    return path


def existing_review_path(batch_dir: Path) -> Path:
    current = review_path(batch_dir)
    return current if current.exists() else Path(batch_dir) / 'review_v2.json'


def read_review(batch_dir: Path) -> dict:
    path = existing_review_path(batch_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('cluster_override', {}), dict):
        raise ValueError('标注格式无效')
    data['reviews'] = clean_reviews(data.get('reviews', {}))
    return data


def write_review(batch_dir: Path, data: dict) -> Path:
    path = review_path(batch_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, 'reviews': clean_reviews(data.get('reviews', {}))}
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix='.review.tmp')
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
