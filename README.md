# MADS Datasets Library

This library provides the functionality to download, process, and stream several datasets.

## Installation
This library has been published on PyPi and can be installed with pip or poetry.

```bash
# Install with pip
pip install mads_datasets

# Install with poetry
poetry add mads_datasets
```

## Data Types
Currently, it supports the following datasets:
* SUNSPOTS Time-Series data
* IMDB Text data
* FLOWERS Image data
* FASHION MNIST Image data
* GESTURES Time-Series data
* IRIS dataset

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

The train.stream() command wil return a generator that will yield batches of data.

You could also create a dataset directly:

```python
dataset = fashion_factory.create_dataset()
```

Or download the data:

```python
fashion_factory.download_data()
```
