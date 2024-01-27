# Copyright © 2023-2024 Apple Inc.
import io
import os
import pickle
import tarfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dataset.cifar10.download_preprocess import dl_preprocess_and_dump


def create_mock_tar_data():
    # Mock tar file in expected format.

    # Create sample data for the pickled objects
    sample_data = {
        b'data': np.ones((1, 32, 32, 3)),
        b'labels': np.zeros((1, ))
    }
    sample_data_bytes = pickle.dumps(sample_data, protocol=4)

    # Create a tar file in memory containing the pickled objects
    tar_data = io.BytesIO()
    with tarfile.open(fileobj=tar_data, mode='w:gz') as tar:
        for file_name in [
                'cifar-10-batches-py/data_batch_1',
                'cifar-10-batches-py/data_batch_2',
                'cifar-10-batches-py/test_batch'
        ]:
            data = io.BytesIO(sample_data_bytes)
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(sample_data_bytes)
            tar.addfile(tarinfo, fileobj=data)
    tar_data.seek(0)
    return tar_data.read()


@pytest.fixture
def mock_endpoint():
    with patch('urllib3.PoolManager') as mock_pool_cls:
        mock_http = MagicMock()
        mock_http.request.return_value = MagicMock(data=create_mock_tar_data())
        mock_pool_cls.return_value = mock_http
        yield


class TestDownloadPreprocess:

    def test_dl_preprocess_and_dump(self, mock_endpoint, tmp_path):
        dl_preprocess_and_dump(tmp_path)
        with open(os.path.join(tmp_path, 'cifar10_train.p'), 'rb') as f:
            images, labels = pickle.load(f)
            np.testing.assert_array_equal(images, np.ones((2, 32, 32, 3)))
            np.testing.assert_array_equal(labels, np.zeros((2, 1)))

        with open(os.path.join(tmp_path, 'cifar10_test.p'), 'rb') as f:
            images, labels = pickle.load(f)
            np.testing.assert_array_equal(images, np.ones((1, 32, 32, 3)))
            np.testing.assert_array_equal(labels, np.zeros((1, 1)))
