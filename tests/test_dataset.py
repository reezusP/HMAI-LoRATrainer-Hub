"""Tests for dataset.py — download, extraction, validation."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset import download_dataset, download_dataset_prefix, extract_zip, validate_dataset


class TestExtractZip:
    def test_flat_structure(self, sample_zip, tmp_path):
        dest = tmp_path / "extracted"
        extract_zip(sample_zip, dest)
        # Should have 10 files (5 images + 5 captions)
        files = list(dest.iterdir())
        assert len(files) == 10

    def test_nested_single_folder(self, sample_zip_nested, tmp_path):
        dest = tmp_path / "extracted"
        extract_zip(sample_zip_nested, dest)
        # Should unwrap the nested folder
        files = list(dest.iterdir())
        assert len(files) == 10
        # No nested subfolder should remain
        dirs = [f for f in dest.iterdir() if f.is_dir()]
        assert len(dirs) == 0

    def test_macosx_skipped(self, sample_zip_macosx, tmp_path):
        dest = tmp_path / "extracted"
        extract_zip(sample_zip_macosx, dest)
        # Should not have __MACOSX folder
        assert not (dest / "__MACOSX").exists()
        files = list(dest.iterdir())
        assert len(files) == 10


class TestValidateDataset:
    def test_all_have_captions(self, sample_dataset_dir):
        unmatched = validate_dataset(sample_dataset_dir)
        assert unmatched == []

    def test_missing_captions(self, sample_dataset_dir_missing_captions):
        unmatched = validate_dataset(sample_dataset_dir_missing_captions)
        assert len(unmatched) == 2
        assert all(name.endswith(".png") for name in unmatched)

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No media files"):
            validate_dataset(empty_dir)

    def test_mixed_extensions(self, sample_dataset_dir_mixed):
        unmatched = validate_dataset(sample_dataset_dir_mixed)
        assert unmatched == []


class TestDownloadDataset:
    def test_successful_download(self, tmp_path):
        dest = tmp_path / "downloaded.zip"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_bytes.return_value = [b"x" * 100]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("dataset.httpx.Client", return_value=mock_client):
            download_dataset("https://example.com/data.zip", dest)
            assert dest.exists()

    def test_http_error(self, tmp_path):
        import httpx

        dest = tmp_path / "fail.zip"
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("dataset.httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                download_dataset("https://example.com/missing.zip", dest)

    def test_corrupt_zip(self, tmp_path):
        corrupt_zip = tmp_path / "corrupt.zip"
        corrupt_zip.write_bytes(b"this is not a zip file")
        dest = tmp_path / "extracted"
        with pytest.raises(zipfile.BadZipFile):
            extract_zip(corrupt_zip, dest)


class TestDownloadDatasetPrefix:
    def _mock_s3(self, keys):
        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": k} for k in keys]}
        ]
        s3.download_file.side_effect = lambda bucket, key, dest: Path(dest).write_bytes(b"x")
        return s3

    def test_downloads_prefix_flat(self, tmp_path):
        keys = [
            "training-datasets/J/images/a.png",
            "training-datasets/J/images/a.txt",
            "training-datasets/J/images/b.jpg",
            "training-datasets/J/images/",  # directory placeholder — skipped
        ]
        s3 = self._mock_s3(keys)
        with patch("uploader._get_s3_client", return_value=(s3, "wan-outputs", "auto")):
            download_dataset_prefix("training-datasets/J/images", tmp_path / "ds")
        files = sorted(f.name for f in (tmp_path / "ds").iterdir())
        assert files == ["a.png", "a.txt", "b.jpg"]

    def test_raises_without_credentials(self, tmp_path):
        with patch("uploader._get_s3_client", return_value=(None, None, None)):
            with pytest.raises(ValueError, match="credentials"):
                download_dataset_prefix("p", tmp_path / "ds")

    def test_raises_on_empty_prefix(self, tmp_path):
        s3 = self._mock_s3([])
        with patch("uploader._get_s3_client", return_value=(s3, "wan-outputs", "auto")):
            with pytest.raises(ValueError, match="No objects"):
                download_dataset_prefix("p", tmp_path / "ds")
