
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.dataset import Dataset_loader



def test_dataloader():
    dataset = Dataset_loader()
    vocab_de, vocab_en = dataset.build_vocab()

    loader = dataset.dataloader(vocab_de, vocab_en, batch_size=8)

    src, tgt = next(iter(loader))

    assert src.shape[1] == 8
    assert tgt.shape[1] == 8

    print("DataLoader test passed.")

test_dataloader()