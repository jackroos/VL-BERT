from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer
import jsonlines
import random
import logging

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Frankie', 'Pat', 'Quinn']


class VCRCorpus(Dataset):
    def __init__(self, ann_file, pretrained_model_name, tokenizer=None, seq_len=64,
                 encoding="utf-8", on_memory=True,
                 **kwargs):
        assert on_memory, "only support on_memory mode!"

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(pretrained_model_name)
        self.vocab = self.tokenizer.vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.ann_file = ann_file
        self.encoding = encoding
        self.test_mode = False

        # load samples into memory
        if on_memory:
            self.corpus = self.load_corpus()

    def load_corpus(self):
        corpus = []
        with jsonlines.open(self.ann_file) as reader:
            person_name_id = 0
            for ann in reader:
                objects_replace_name = []
                for o in ann['objects']:
                    if o == 'person':
                        objects_replace_name.append(GENDER_NEUTRAL_NAMES[person_name_id])
                        person_name_id = (person_name_id + 1) % len(GENDER_NEUTRAL_NAMES)
                    else:
                        objects_replace_name.append(o)
                question = self.convert_mixed_tokens_to_natural_sent(ann['question'], objects_replace_name)
                answer_choices = [self.convert_mixed_tokens_to_natural_sent(choice, objects_replace_name)
                                  for choice in ann['answer_choices']]
                rationale_choices = [self.convert_mixed_tokens_to_natural_sent(choice, objects_replace_name)
                                     for choice in ann['rationale_choices']]
                corpus.append((question, answer_choices[ann['answer_label']], rationale_choices[ann['rationale_label']]))

        return corpus

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        q, a, r = self.corpus[item]

        # tokenize
        tokens_q = self.tokenizer.tokenize(q)
        tokens_a = self.tokenizer.tokenize(a)
        tokens_r = self.tokenizer.tokenize(r)
        tokens = tokens_q + tokens_a + tokens_r

        # masked language modeling
        tokens, mlm_labels = self.random_word(tokens)

        # convert token to its vocab id
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]

        return ids, mlm_labels

    @staticmethod
    def convert_mixed_tokens_to_natural_sent(tokens, objects_replace_name):
        parsed_tokens = []
        for mixed_token in tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                parsed_tokens.append(tokens[0])
                for token in tokens[1:]:
                    parsed_tokens.append('and')
                    parsed_tokens.append(token)
            else:
                parsed_tokens.append(mixed_token)

        sent = ' '.join(parsed_tokens)
        return sent

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tokenizer.vocab["[UNK]"])
                    logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        # if no word masked, random choose a word to mask
        if all([l_ == -1 for l_ in output_label]):
            choosed = random.randrange(0, len(output_label))
            output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return tokens, output_label
