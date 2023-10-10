# MADS Datasets Library

This library provides the functionality to download, process, and stream several datasets.

## Installation
This library has been published on PyPi and can be installed with pip, conda, pdm or poetry.

```bash
# Install with pip
pip install mads_datasets

# Install with poetry
poetry add mads_datasets

# install with pdm
pdm add mads_datasets
```

## Data Types
Currently, it supports the following datasets:
* SUNSPOTS Time-Series data, 3000 monthly sunspot observations from 1749
* IMDB Text data, 50k movie reviews with positive and negative sentiment labels
* FLOWERS Image data, about 3000 large and complex images of 5 flowers
* FASHION MNIST Image data, 60k images sized 28x28 pixels
* GESTURES Time-Series data with x, y and z accelerometer data for 20 gestures.
* IRIS dataset, 150 observations of 4 features of 3 iris flower species
* PENGUINS dataset, an alternative to Iris with 344 penguins on multiple islands.
* FAVORITA dataset, 125 million sales records of 50k products in 54 stores.

An additional type is SECURE, which is used for datasets that are not publicly available. See examples below.

## Usage

After installation, import the necessary components:

```python
from mads_datasets import DatasetFactoryProvider, DatasetType
```

You can create a specific dataset factory using the `DatasetFactoryProvider`.

For instance, to create a factory for the Fashion MNIST dataset:

```python
fashion_factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
```

With the factory, you can download the data, create datasets and provide the datasets wrapped in datastreamers in one command:

```python
streamers = mnistfactory.create_datastreamer(batchsize=32)
train = streamers["train"]
X, y = next(train.stream())
```

This will do multiple things:
- it downloads the data to the default location `~/.cache/mads_datasets` if not already present
- it checks the hash of the file
- it transforms the file into a dataset that implements the DatasetProtocol:
```python
class DatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Any:
        ...
```
- it creates a streamer.

The train.stream() command wil return a generator that yields batches of data.

You could also create a dataset directly:

```python
dataset = fashion_factory.create_dataset()
```

Or just download the data:

```python
fashion_factory.download_data()
```

## Examples

### Default cache location
```python
fashion_factory.datadir
```

This shows you the default location of all datasets. Storing the datasets in a central place makes
it much easier to manage your storage over multiple projects where you might want to reuse the
same dataset multiple times.

It also makes it much easier to clean you storage, as you can just delete the entire directory.

In general, it is bad practice to store the raw data in your git repo. Using a central location avoids this (as would adding your `data` folder to `.gitignore`).
Instead of storing data in git, provide the user
with a script
- that downloads the raw data
- preprocesses the raw data into something that fits your needs

If you want to change the default location of the cache, give the factory a `datadir` argument like this:

```python
from pathlib import Path
fashion_factory = DatasetFactoryProvider.create_factory(
    dataset_type=DatasetType.FASHION,
    datadir=Path("~/path/to/alternative/folder")
    )
```
Now, instead of using `~/.cache/mads_datasets`, the data will be stored in `~/path/to/alternative/folder` which can be anywhere you have access to.

### using Secure datasets

```
from mads_datasets.settings import SecureDatasetSettings
from mads_datasets import DatasetFactoryProvider, DatasetType

garbagesettings = SecureDatasetSettings(
    dataset_url="https://gitlab.com/api/v4/projects/12345/repository/files/filename.extension/raw?lfs=true",
    filename="garbage.parq",
    name="garbage",
    keyaccount="gitlab-MADS-PAT",
    keyname="gitlab-MADS-PAT",
    digest="b5ee4ab8723e0d97e0eefa12e347d04e",
    unzip=False,
)
garbagedata = DatasetFactoryProvider.create_factory(settings=garbagesettings, dataset_type=DatasetType.SECURE)
garbagedata.download_data()
```

For secure datasets, you will have to provide a `SecureDatasetSettings` object. Currently, this is implemented for the gitlab API.

You will have to provide an url where you change 12345 with the project ID (settings>general). You should also create a PAT (settings> access tokens).

Change `filename.extension` with your filename (eg. `garbage.parq`).
The first time you run this, you will have to provide your PAT. It will be stored in your keyring as `gitlab-MADS-PAT`, or whatever you set as `keyaccount` and `keyname`.

The digest can be left out; the first time you run this, you will get the digest and you can add it to make sure the file is not corrupted.

If your file needs unzipping, set `unzip=True`.

