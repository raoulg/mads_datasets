from mads_datasets import tokenizer


def test_tokenize():
    text = "This is a test."
    vocab = tokenizer.build_vocab(text, max=2)