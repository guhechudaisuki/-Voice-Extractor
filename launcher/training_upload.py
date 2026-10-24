"""Bound material uploads before multipart files are spooled to disk."""
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

MAX_UPLOAD_BYTES = 8 * 1024**3 + 1024**2  # Includes multipart headers.


class MaterialUploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/api/training/start':
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_UPLOAD_BYTES:
            response = JSONResponse({'detail': '原料上传总大小超过 8 GB 或请求长度无效'}, status_code=413)
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > MAX_UPLOAD_BYTES:
                # Starlette catches this during parsing and closes every spool file.
                raise MultiPartException('原料上传总大小超过 8 GB')
            return message

        await self.app(scope, limited_receive, send)
