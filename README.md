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
