"""Stable query/reference sample identities for universal learning and replay."""
import hashlib
import json
from pathlib import Path
from extractor.model_store import file_sha256

def role_hashes(item):
    return sorted(item.get('role_reference_sha256') or [file_sha256(Path(p)) for p in item.get('reference_paths',())])

def sample_fingerprint(item):
    return hashlib.sha256(json.dumps([item.get('source_sha256') or sorted(item['source_groups']),
        item['sha256'], role_hashes(item) or item.get('reference_key'),
        item['review']['label']],separators=(',',':')).encode()).hexdigest()

