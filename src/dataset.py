import numpy as np 
import torch 

from torchtext.datasets import Multi30k
import spacy 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import spacy



# for i,sample in enumerate(train_iter):
#     print(sample)
#     if i == 12:
#         break

tokenizer_de = get_tokenizer("spacy", language="de_core_news_sm")
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")

class Dataset_loader():

    def __init__(self):
        self.vocab_de = None
        self.vocab_en = None
    

    def yield_token(self,data_iter,language):
        for de_sent,en_sent in data_iter:
            if language == "de":
                yield tokenizer_de(de_sent)
            else:
                yield tokenizer_en(en_sent)
    

    def numericalize(self,vocab, tokenizer, text):
        tokens = tokenizer(text)
        ids = vocab(tokens)
        return [vocab["<sos>"]] + ids + [vocab["<eos>"]]
    

    def collate_fn(self,batch,vocab_de, vocab_en):

        src_batch, tgt_batch = [], []

        for de_sent, en_sent in batch:
            src_ids = self.numericalize(vocab_de, tokenizer_de, de_sent)
            tgt_ids = self.numericalize(vocab_en, tokenizer_en, en_sent)

            src_batch.append(torch.tensor(src_ids))
            tgt_batch.append(torch.tensor(tgt_ids))

        src_batch = pad_sequence(src_batch, padding_value=vocab_de["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=vocab_en["<pad>"])

        return src_batch, tgt_batch
        
    def build_vocab(self):

        self.train_iter = Multi30k(split='train', language_pair=('de', 'en'))
        self.vocab_de = build_vocab_from_iterator(self.yield_token(self.train_iter,"de"), specials=["<sos>", "<eos>", "<unk>", "<pad>"])
        self.train_iter = Multi30k(split='train', language_pair=('de', 'en'))
        self.vocab_en = build_vocab_from_iterator(self.yield_token(self.train_iter,"en"), specials=["<sos>", "<eos>", "<unk>", "<pad>"])


        self.vocab_de.set_default_index(self.vocab_de["<unk>"])
        self.vocab_en.set_default_index(self.vocab_en["<unk>"])

        return self.vocab_de, self.vocab_en

    def dataloader(self, vocab_de, vocab_en , batch_size=32):
        
        
        train_iter = Multi30k(split='train', language_pair=('de', 'en'))

        return DataLoader(
            list(train_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.collate_fn(batch, vocab_de, vocab_en)

        )

   
    # train_iter = Multi30k(split='train', language_pair=('de', 'en'))

    # # first_sample = next(iter(train_iter))

    # # de_sent,en_sent = first_sample

    # # print(f"German sentence: {de_sent}")
    # # print(f"English sentence: {en_sent}")


    # # print("\nTokenized German sentence:")
    # # print(tokenizer_de(de_sent))
    # # print("\nTokenized English sentence:")
    # # print(tokenizer_en(en_sent))

    # total_samples = sum(1 for _ in train_iter)
    # print(f"\nTotal number of samples in the training set: {total_samples}")


    # Build vocabulary from the training iterator

    

    

    # Re-initialize the training iterator since it was exhausted
    # train_iter = Multi30k(split='train', language_pair=('de', 'en'))

    # vocab_de = build_vocab_from_iterator(yield_token(train_iter,"de"), specials=["<sos>", "<eos>", "<unk>", "<pad>"])


    # train_iter = Multi30k(split='train', language_pair=('de', 'en'))

    # vocab_en = build_vocab_from_iterator(yield_token(train_iter,"en"), specials=["<sos>", "<eos>", "<unk>", "<pad>"])

    # vocab_de.set_default_index(vocab_de["<unk>"])
    # vocab_en.set_default_index(vocab_en["<unk>"])

    # # Example of numericalization
    # sample_sentence = "Ein Mann isst eine Pizza."
    # numericalized = numericalize(vocab_de, tokenizer_de, sample_sentence)
    # print(f"\nNumericalized German sentence: {numericalized}")






    

    
    

   


















    











