import random
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer
import logging


class GeneralCorpus(Dataset):
    def __init__(self, ann_file, pretrained_model_name, tokenizer=None, seq_len=64, min_seq_len=64,
                 encoding="utf-8", on_memory=True,
                 **kwargs):
        assert on_memory, "only support on_memory mode!"

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(pretrained_model_name)
        self.vocab = self.tokenizer.vocab
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.on_memory = on_memory
        self.ann_file = ann_file
        self.encoding = encoding
        self.test_mode = False

        # load samples into memory
        if on_memory:
            self.corpus = self.load_corpus()

    def load_corpus(self):
        corpus = []
        for ann_file in self.ann_file.split('+'):
            with open(ann_file, 'r', encoding=self.encoding) as f:
                corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in f.readlines()])

        corpus = [l.strip() for l in corpus if l.strip() != '']

        return corpus

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        raw = self.corpus[item]

        # tokenize
        tokens = self.tokenizer.basic_tokenizer.tokenize(raw)

        # add more tokens if len(tokens) < min_len
        _cur = (item + 1) % len(self.corpus)
        while len(tokens) < self.min_seq_len:
            _cur_tokens = self.tokenizer.basic_tokenizer.tokenize(self.corpus[_cur])
            tokens.extend(_cur_tokens)
            _cur = (_cur + 1) % len(self.corpus)

        # masked language modeling
        tokens, mlm_labels = self.random_word_wwm(tokens)

        # convert token to its vocab id
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]

        return ids, mlm_labels

    # def random_word(self, tokens):
    #     output_label = []
    #
    #     for i, token in enumerate(tokens):
    #         prob = random.random()
    #         # mask token with 15% probability
    #         if prob < 0.15:
    #             prob /= 0.15
    #
    #             # 80% randomly change token to mask token
    #             if prob < 0.8:
    #                 tokens[i] = "[MASK]"
    #
    #             # 10% randomly change token to random token
    #             elif prob < 0.9:
    #                 tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
    #
    #             # -> rest 10% randomly keep current token
    #
    #             # append current token to output (we will predict these later)
    #             try:
    #                 output_label.append(self.tokenizer.vocab[token])
    #             except KeyError:
    #                 # For unknown words (should not occur with BPE vocab)
    #                 output_label.append(self.tokenizer.vocab["[UNK]"])
    #                 logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
    #         else:
    #             # no masking token (will be ignored by loss function later)
    #             output_label.append(-1)
    #
    #     # if no word masked, random choose a word to mask
    #     if self.force_mask:
    #         if all([l_ == -1 for l_ in output_label]):
    #             choosed = random.randrange(0, len(output_label))
    #             output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]
    #
    #     return tokens, output_label

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        ## if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label