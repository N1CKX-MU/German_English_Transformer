from src.dataset import Dataset_loader

def test_vocab_build():
    dataset = Dataset_loader()
    vocab_de, vocab_en = dataset.build_vocab()

    assert "<sos>" in vocab_de.get_stoi()
    assert "<eos>" in vocab_en.get_stoi()
    assert len(vocab_de) > 0
    assert len(vocab_en) > 0

    print("Vocab build test passed.")