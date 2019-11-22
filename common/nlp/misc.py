import torch
import random


def get_align_matrix(aligned_ids, sparse=False, device=None, dtype=torch.float32):
    """
    Get aligned matrix for feature alignment in sentence embedding
    :param aligned_ids: list, aligned_ids[k] means original index of k-th token
    :param sparse: whether to return sparse matrix
    :param device: device of returned align matrix
    :param dtype: dtype of returned align matrix
    :return: align_matrix: torch.FloatTensor, shape: (L, L')

    Example:
    >> aligned_ids = [0, 0, 1, 2, 2, 2]
    >> get_align_matrix(aligned_ids)
    tensor([[0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.3333]])
    """

    l0 = max(aligned_ids) + 1
    l1 = len(aligned_ids)
    if sparse:
        raise NotImplementedError
    else:
        align_matrix = torch.zeros((l0, l1), dtype=dtype, device=device)
        align_matrix[aligned_ids, torch.arange(l1)] = 1
        align_matrix = align_matrix / align_matrix.sum(dim=1, keepdim=True)

    return align_matrix


def get_all_ngrams(words):
    """
    Get all n-grams of words
    :param words: list of str
    :return: ngrams, list of (list of str)
    """
    ngrams = []
    N = len(words)
    for n in range(1, N + 1):
        for i in range(0, N - n + 1):
            ngrams.append([words[j] for j in range(i, i + n)])

    return ngrams


def random_word_with_token_ids(token_ids, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param token_ids: list of int, list of token id.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    for i, token_id in enumerate(token_ids):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                token_ids[i] = mask_id

            # 10% randomly change token to random token
            elif prob < 0.9:
                token_ids[i] = random.choice(list(tokenizer.vocab.items()))[1]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token_id)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return token_ids, output_label





