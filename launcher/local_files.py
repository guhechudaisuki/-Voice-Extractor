"""Bounded local file selection; network clients cannot enumerate the host."""
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

MEDIA = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.opus', '.mp4', '.mkv', '.mov', '.avi', '.webm', '.wma'}


def require_loopback(request):
    from fastapi import HTTPException
    host = request.client.host if request.client else ''
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    url_host = request.url.hostname
    if not local or url_host not in {'localhost', '127.0.0.1', '::1'}:
        raise HTTPException(403, '文件浏览仅限本机访问')
    origin = request.headers.get('origin')
    if origin and urlsplit(origin).netloc != request.url.netloc:
        raise HTTPException(403, '拒绝跨站文件浏览请求')


def browse(path='', *, kind='media', offset=0, limit=200):
    if kind not in {'media', 'subtitle', 'directory'} or offset < 0 or not 1 <= limit <= 500:
        raise ValueError('浏览参数无效')
    if not path:
        roots = [Path(f'{letter}:\\') for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'] if os.name == 'nt' else [Path('/')]
        return {'path': '', 'parent': None, 'entries': [{'name': str(p), 'path': str(p), 'directory': True}
                  for p in roots if p.is_dir()], 'next_offset': None}
    folder = Path(path).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('请选择目录')
    extensions = {'.srt', '.ass', '.ssa', '.vtt', '.txt'} if kind == 'subtitle' else MEDIA
    entries = []
    # Stop enumeration at a bounded page, even in huge directories.
    more = False
    count = 0
    with os.scandir(folder) as iterator:
        for item in iterator:
            try:
                is_dir = item.is_dir()
                if not is_dir and (kind == 'directory' or Path(item.name).suffix.lower() not in extensions):
                    continue
                if count >= offset:
                    if len(entries) == limit:
                        more = True
                        break
                    entries.append({'name': item.name, 'path': str(Path(item.path).resolve()), 'directory': is_dir})
                count += 1
            except OSError:
                continue
    entries.sort(key=lambda e: (not e['directory'], e['name'].casefold()))
    return {'path': str(folder), 'parent': str(folder.parent) if folder.parent != folder else '',
            'entries': entries, 'next_offset': offset + limit if more else None}
