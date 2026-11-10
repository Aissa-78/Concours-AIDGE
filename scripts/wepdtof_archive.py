"""Lecture partielle d'un ZIP public par HTTP Range, sans charger toute l'archive.

Le serveur doit confirmer chaque plage demandée. Pas de repli sur un
téléchargement intégral si les requêtes partielles ne sont pas acceptées.
"""

from __future__ import annotations

import io
import re

import requests


class RangeArchive(io.RawIOBase):
    def __init__(self, url, max_bytes=256 * 1024 * 1024, session=None):
        super().__init__()
        self.url = url
        self.session = session or requests.Session()
        self.owns_session = session is None
        self.position = 0
        self.transferred = 0
        self.max_bytes = max_bytes
        self.etag = None
        with self.session.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                              stream=True, timeout=(20, 60)) as response:
            if response.status_code != 206:
                raise ValueError("Le serveur ne permet pas la lecture partielle (HTTP 206 attendu).")
            match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise ValueError("Taille de l'archive non confirmée par le serveur.")
            self.size = int(match[1])
            self.etag = response.headers.get("ETag")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Origine de seek invalide.")
        if position < 0:
            raise ValueError("Position négative dans le ZIP.")
        self.position = position
        return position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Archive fermée.")
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        if size > 64 * 1024 * 1024 or self.transferred + size > self.max_bytes:
            raise ValueError("Limite de transfert atteinte ; l'archive complète ne sera pas téléchargée.")
        start, end = self.position, self.position + size - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                   "Cache-Control": "no-cache"}
        if self.etag:
            headers["If-Range"] = self.etag
        for attempt in range(3):
            try:
                with self.session.get(self.url, headers=headers, stream=True, timeout=(20, 60)) as response:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Le serveur n'a pas retourné la plage exacte demandée.")
                    if self.etag and response.headers.get("ETag") != self.etag:
                        raise ValueError("L'archive distante a changé pendant la lecture.")
                    data = response.raw.read(size + 1)
                    self.transferred += len(data)
                    if len(data) != size:
                        raise ValueError("Plage HTTP tronquée ou trop longue.")
                    self.position += size
                    return data
            except requests.RequestException:
                if attempt == 2:
                    raise

    def close(self):
        if not self.closed and self.owns_session:
            self.session.close()
        super().close()
