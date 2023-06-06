import re
import string
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING, List, Tuple

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab

from mads_datasets.datasets import TextDataset
from mads_datasets.settings import TextDatasetSettings

if TYPE_CHECKING:
    from mads_datasetfactory import PreprocessingProtocol

Tensor = torch.Tensor


def clean(text: str) -> str:
    punctuation = f"[{string.punctuation}]"
    # remove CaPiTaLs
    lowercase = text.lower()
    # change don't and isn't into dont and isnt
    neg = re.sub("\\'", "", lowercase)
    # swap html tags for spaces
    html = re.sub("<br />", " ", neg)
    # swap punctuation for spaces
    stripped = re.sub(punctuation, " ", html)
    # remove extra spaces
    spaces = re.sub("  +", " ", stripped)
    return spaces


class BaseTokenizer(PreprocessingProtocol):
    def __init__(
        self, traindataset: TextDataset, settings: TextDatasetSettings
    ) -> None:
        self.maxvocab = settings.maxvocab
        self.maxtokens = settings.maxtokens
        self.clean = settings.clean_fn
        self.vocab = self.build_vocab(self.build_corpus(traindataset))

    @staticmethod
    def split_and_flat(corpus: List[str]) -> List[str]:
        """
        Split a list of strings on spaces into a list of lists of strings
        and then flatten the list of lists into a single list of strings.
        eg ["This is a sentence"] -> ["This", "is", "a", "sentence"]
        """
        corpus_ = [x.split() for x in corpus]
        corpus = [x for y in corpus_ for x in y]
        return corpus

    def build_corpus(self, dataset) -> List[str]:
        corpus = []
        for i in range(len(dataset)):
            x = self.clean(dataset[i][0])
            corpus.append(x)
        return corpus

    def build_vocab(
        self, corpus: List[str], oov: str = "<OOV>", pad: str = "<PAD>"
    ) -> Vocab:
        data = self.split_and_flat(corpus)
        counter = Counter(data).most_common()
        logger.info(f"Found {len(counter)} tokens")
        counter = counter[: self.maxvocab - 2]
        ordered_dict = OrderedDict(counter)
        v1 = vocab(ordered_dict, specials=[pad, oov])
        v1.set_default_index(v1[oov])
        return v1

    def cast_label(self, label: str) -> int:
        raise NotImplementedError

    def __call__(self, batch: List) -> Tuple[Tensor, Tensor]:
        labels, text = [], []
        for x, y in batch:
            if clean is not None:
                x = self.clean(x)  # type: ignore
            x = x.split()[: self.maxtokens]
            tokens = torch.tensor([self.vocab[word] for word in x], dtype=torch.int32)
            text.append(tokens)
            labels.append(self.cast_label(y))

        text_ = pad_sequence(text, batch_first=True, padding_value=0)
        return text_, torch.tensor(labels)


class IMDBTokenizer(BaseTokenizer):
    def __init__(self, traindataset, settings):
        super().__init__(traindataset, settings)

    def cast_label(self, label: str) -> int:
        if label == "neg":
            return 0
        else:
            return 1
