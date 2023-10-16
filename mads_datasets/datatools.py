from __future__ import annotations

import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
from keyring.errors import NoKeyringError

import keyring
import requests
import os
from loguru import logger
from tqdm import tqdm

from mads_datasets.settings import FileTypes, SecureDatasetSettings

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor


def walk_dir(path: Path) -> Iterator:
    """loops recursively through a folder

    Args:
        path (Path): folder to loop trough. If a directory
            is encountered, loop through that recursively.

    Yields:
        Generator: all paths in a folder and subdirs.
    """

    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk_dir(p)
            continue
        # resolve works like .absolute(), but it removes the "../.." parts
        # of the location, so it is cleaner
        yield p.resolve()


def iter_valid_paths(
    path: Path, formats: List[FileTypes]
) -> Tuple[Iterator, List[str]]:
    """
    Gets all paths in folders and subfolders
    strips the classnames assuming that the subfolders are the classnames
    Keeps only paths with the right suffix


    Args:
        path (Path): image folder
        formats (List[str]): suffices to keep, eg [".jpg", ".png"]

    Returns:
        Tuple[Iterator, List[str]]: _description_
    """
    # gets all files in folder and subfolders
    walk = walk_dir(path)
    # retrieves foldernames as classnames
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    # keeps only specified formats
    formats_ = [f.value for f in formats]
    paths = (path for path in walk if path.suffix in formats_)
    return paths, class_names


def get_file(
    data_dir: Path,
    filename: Path,
    url: str,
    unzip: bool = True,
    overwrite: bool = False,
    headers: dict = None,
) -> Path:
    """Download a file from url to location data_dir / filename

    Args:
        data_dir (Path): dir to store file
        filename (Path): filename
        url (str): url to obtain filename
        unzip (bool, optional): If the file needs unzipping
        overwrite (bool, optional): overwrite file, if it already exists.

    Returns:
        Path: _description_
    """

    path = data_dir / filename
    if path.exists() and not overwrite:
        logger.info(f"File {path} already exists, skip download")
        return path
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}")
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 2**10
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, colour="#1e4706"
    )
    logger.info(f"Downloading {path}")
    with open(path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if unzip:
        extract(path)
    return path


def extract(path: Path) -> None:
    try:
        file_type = FileTypes(path.suffix)
    except ValueError:
        logger.info(f"The suffix {path.suffix} is not a recognized file type.")

    if file_type in [FileTypes.ZIP]:
        logger.info(f"Unzipping {path}")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)

    if file_type in [FileTypes.TGZ, FileTypes.TAR, FileTypes.GZ]:
        logger.info(f"Unzipping {path}")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=path.parent)


def clean_dir(dir: Union[str, Path]) -> None:
    dir = Path(dir)
    if dir.exists():
        logger.info(f"Clean out {dir}")
        shutil.rmtree(dir)
    else:
        dir.mkdir(parents=True)


def window(x: "Tensor", n_time: int) -> "Tensor":
    """
    Generates and index that can be used to window a timeseries.
    E.g. the single series [0, 1, 2, 3, 4, 5] can be windowed into 4 timeseries with
    length 3 like this:

    [0, 1, 2]
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]

    We now can feed 4 different timeseries into the model, instead of 1, all
    with the same length.
    """
    with import_torch() as torch:  # type: ignore
        n_window = len(x) - n_time + 1
        time = torch.arange(0, n_time).reshape(1, -1)  # type: ignore
        window = torch.arange(0, n_window).reshape(-1, 1)  # type: ignore
        idx = time + window
        return idx


def dir_add_timestamp(log_dir: Optional[Path] = None) -> Path:
    if log_dir is None:
        log_dir = Path(".")
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = log_dir / timestamp
    logger.info(f"Logging to {log_dir}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    return log_dir


def keep_subdirs_only(path: Path) -> None:
    files = [file for file in path.iterdir() if file.is_file()]
    for file in files:
        file.unlink()


@contextmanager
def import_torch() -> Optional["torch"]:  # type: ignore
    try:
        import torch
    except ImportError:
        logger.warning(
            "Torch is not installed."
            "Please install 'mads-datasets[torch]' to use this feature."
        )
        yield None
    else:
        yield torch


def check_token(account: str, name: str) -> str:
    haskeyring = True
    notstored = False
    try:
        token = keyring.get_password(account, name)
        if not token:
            notstored = True
    except NoKeyringError as e:
        haskeyring = False
        logger.warning(
            f"Failed to find a keyring backend: {e}"
            f"Falling back to environment to obtain token from variable {account}."
        )
        token = os.environ.get(account)
        if not token:
            logger.warning(
                f"Could not obtain {account} from the environment."
                "Please add this to your environment"
            )

    if not token:
            logger.info("Enter your token for {account} and {name} manually:")
            token = input("Token: ")
    if haskeyring and notstored:
        logger.info(f"Storing token for {account} and {name} in keyring.")
        keyring.set_password(account, name, token)
    return token


def create_headers(settings: SecureDatasetSettings) -> dict:
    token = check_token(settings.keyaccount, settings.keyname)

    headers = {
        "PRIVATE-TOKEN": token,
    }
    return headers
