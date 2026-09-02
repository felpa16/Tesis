"""Minimal S3 client shared by the feature extractor and the analysis scripts.

Kept free of torch and transformers so a plotting or inspection script can talk
to the bucket without paying a model import.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np


class S3:
    """Thin boto3 wrapper handing each thread its own client.

    boto3 clients are not documented as thread-safe, and this script uploads
    from a worker pool while the main thread runs MERT.
    """

    def __init__(self, bucket: str, region: str | None = None) -> None:
        import boto3

        self.bucket = bucket
        self._session = boto3.session.Session(region_name=region)
        self._local = threading.local()

    @property
    def client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._session.client("s3")
            self._local.client = client
        return client

    def list_keys(self, prefix: str) -> list[str]:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", ()):
                if not obj["Key"].endswith("/") and obj["Size"] > 0:
                    keys.append(obj["Key"])
        return keys

    def put_array(self, key: str, array: np.ndarray) -> int:
        buffer = io.BytesIO()
        np.save(buffer, array)
        size = buffer.tell()
        buffer.seek(0)
        self.client.upload_fileobj(buffer, self.bucket, key)
        return size

    def put_file(self, key: str, path: Path) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def get_file(self, key: str, path: Path) -> None:
        self.client.download_file(self.bucket, key, str(path))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def folders(self, prefix: str = "") -> list[str]:
        """Immediate 'subdirectories' of a prefix, for diagnosing key layout."""
        page = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/", MaxKeys=200
        )
        return [p["Prefix"] for p in page.get("CommonPrefixes", ())]

    def sample(self, prefix: str, limit: int = 3) -> list[str]:
        page = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=limit
        )
        return [o["Key"] for o in page.get("Contents", ())]

    def get_array(self, key: str) -> np.ndarray:
        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buffer)
        buffer.seek(0)
        return np.load(buffer)
