"""Retain the target references that give a reviewed clip's label its meaning."""
import json
from pathlib import Path
import shutil

from .model_store import file_sha256


def save_training_context(output: Path, source: Path, references: list[Path]):
    directory = output / '_training'
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for reference in references:
        checksum = file_sha256(reference)
        name = checksum + reference.suffix.lower()
        shutil.copyfile(reference, directory / name)
        records.append({'audio': name, 'sha256': checksum})
    data = {'schema': 1, 'source_name': source.name, 'source_sha256': file_sha256(source),
            'references': records}
    (directory / 'context.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load_training_context(directory: Path) -> dict:
    marker = directory / '_training/context.json'
    if marker.is_file():
        data = json.loads(marker.read_text(encoding='utf-8'))
        refs = []
        for item in data.get('references', []):
            path = (marker.parent / item['audio']).resolve()
            if path.parent != marker.parent.resolve() or not path.is_file() or file_sha256(path) != item['sha256']:
                raise ValueError('批次参考音频缺失或校验失败，请重新提取')
            refs.append(path)
        if refs:
            return {**data, 'reference_paths': refs}
    # Historical batches recorded the reference paths and their digests in the
    # cluster provenance. Use them only when the exact original files survive.
    report_path = directory / 'cluster_report.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        provenance = report.get('provenance') or {}
        digests = {item['path']: item['sha256'] for item in provenance.get('inputs', [])}
        references = provenance.get('reference_paths') or []
        if references and all(Path(path).is_file() and digests.get(path) == file_sha256(Path(path)) for path in references):
            return {'reference_paths': list(map(Path, references)),
                    'source_name': (report.get('meta') or {}).get('source_name', directory.name),
                    'source_sha256': provenance.get('source_sha256')}
    raise ValueError('此旧批次缺少可核对的目标参考音频，无法导出训练原料；请重新提取该批次。普通音频 ZIP 仍可导出')
