from typing import List, Tuple
from pathlib import Path
import pytest
from mads_datasets.datatools import iter_valid_paths, get_file
from mads_datasets.settings import FileTypes
import responses
from unittest import mock

class TestData:
    @pytest.fixture
    def mock_directory(self, tmpdir_factory) -> Tuple[Path, List[str], List[str]]:
        dir = Path(tmpdir_factory.mktemp("data").strpath)
        formats = [".jpg", ".png"]
        classes = ["class1", "class2"]
        for cls in classes:
            for i in range(2):
                for format_ in formats:
                    file = dir / cls / f"image{i}{format_}"
                    file.parent.mkdir(parents=True, exist_ok=True)
                    with open(file, "w") as f:
                        f.write("test")
        return dir, classes, formats

    def test_iter_valid_paths(self, mock_directory):
        path, expected_classes, formats = mock_directory
        paths, classes = iter_valid_paths(Path(path), [FileTypes.JPG, FileTypes.PNG])

        # Check if classes are correctly recognized
        assert sorted(classes) == sorted(expected_classes)

        # Check if all paths are returned and they have correct formats
        returned_files = list(paths)
        assert len(returned_files) == len(expected_classes) * 2 * len(formats)
        for path in returned_files:
            assert path.suffix in formats

        # Check if files belong to correct directories (classes)
        for path in returned_files:
            assert path.parent.name in expected_classes


@responses.activate
def test_get_file(tmpdir):
    # Setup
    data_dir = Path(tmpdir)
    filename = "file.txt"
    url = "http://example.com/"
    path = data_dir / filename

    responses.add(responses.GET, url, body=b"content")

    with mock.patch.object(Path, "exists", return_value=False):
        with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            with mock.patch("mads_datasets.datatools.extract") as mock_extract:
                # Call the function
                result = get_file(data_dir, filename, url, unzip=True, overwrite=False)

    # Check the results
    assert result == path
    mock_file.assert_called_once_with(path, "wb")
    mock_extract.assert_called_once_with(path)

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == url

@responses.activate
def test_get_file_already_exists(tmpdir):
    # Setup
    data_dir = Path(tmpdir)
    filename = "file.txt"
    url = "http://example.com"
    path = data_dir / filename

    responses.add(responses.GET, url, body=b"content")

    with mock.patch.object(Path, "exists", return_value=True):
        # Call the function
        result = get_file(data_dir, filename, url, unzip=True, overwrite=False)

    # Check the results
    assert result == path
    assert len(responses.calls) == 0  # The file already exists, so no request should be made
