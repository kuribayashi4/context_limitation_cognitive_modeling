import argparse
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
from fairseq.models.lstm_lm import LSTMLanguageModel
from fairseq.models.transformer_lm import TransformerLanguageModel
from torch.nn import CrossEntropyLoss


def concat_bos(tensor, bos):
    return torch.cat([torch.tensor([bos]), tensor])


@torch.no_grad()
def batched_decode(custom_lm, texts, target_spans, loss_fct, device, args):

    random.seed(0)
    pad_id = custom_lm.src_dict.pad()
    bos = custom_lm.src_dict.bos()
    batchsize = args.batchsize

    assert len(texts) == len(target_spans)
    texts_target_spans = [
        (t_s[0], t_s[1], i) for i, t_s in enumerate(zip(texts, target_spans))
    ]
    texts_target_spans = sorted(texts_target_spans, key=lambda x: len(x[0].split()))

    sorted_texts = list(map(lambda x: x[0], texts_target_spans))
    sorted_target_spans = list(map(lambda x: x[1], texts_target_spans))
    sorted_indices = list(map(lambda x: x[2], texts_target_spans))

    # eos is added at the end of input
    sorted_input_ids = [custom_lm.binarize(t)[:-2] for t in sorted_texts]
    sorted_gold_ids = [custom_lm.binarize(t)[1:-1] for t in sorted_texts]

    surprisals4pieces = []
    surprisals4bunsetsu = [0] * len(sorted_indices)
    for i in range(math.ceil(len(texts) / batchsize)):
        batched_input_ids = sorted_input_ids[batchsize * i : batchsize * (i + 1)]
        batched_gold_ids = sorted_gold_ids[batchsize * i : batchsize * (i + 1)]
        batched_target_spans = sorted_target_spans[batchsize * i : batchsize * (i + 1)]
        batched_ids = sorted_indices[batchsize * i : batchsize * (i + 1)]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batched_input_ids, batch_first=True, padding_value=pad_id
        )
        padded_gold_ids = torch.nn.utils.rnn.pad_sequence(
            batched_gold_ids, batch_first=True, padding_value=pad_id
        )

        # SHAPE: (batchsize, max_length, vocab_size)
        results = custom_lm.models[0](padded_input_ids.to(device))[0]

        # TODO: avoid for loop
        for result, gold_ids, span, bunsetsu_id in zip(
            results, padded_gold_ids, batched_target_spans, batched_ids
        ):
            # SHAPE: max_length
            positional_scores = loss_fct(result, gold_ids.to(device))

            # target側はindexが一つずれているため，spanのindexも1つ前にずらす
            surprisals = (
                positional_scores[span[0] - 1 : span[1]]
                .cpu()
                .detach()
                .clone()
                .numpy()
                .tolist()
            )

            # TODO: change base
            bunsetsu_surprisal = sum(surprisals)
            assert not bunsetsu_surprisal < 0
            surprisals4bunsetsu[bunsetsu_id] = bunsetsu_surprisal
            surprisals4pieces.extend(surprisals)
    return surprisals4bunsetsu, surprisals4pieces


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.corpus == "BCCWJ":
        DATA_PATH = "data/ja_sents"
        SPM_PATH = "models/spm_ja/japanese_gpt2_unidic"
        article2piece = json.load(open(args.data_path))
    elif args.corpus == "dundee":
        DATA_PATH = "data/en_sents"
        SPM_PATH = "models/spm_en/en_wiki"
        article2piece = json.load(open(args.data_path))
    else:
        raise NotImplementedError()

    article2scores = defaultdict(lambda: defaultdict(list))
    print(args.model_path)
    if args.arch in ["transformer", "gpt"]:
        custom_lm = TransformerLanguageModel.from_pretrained(
            os.path.dirname(args.model_path),
            os.path.basename(args.model_path),
            data_name_or_path=DATA_PATH + "/data-bin-ngram"
            if args.add_itag
            else DATA_PATH + "/data-bin",
            bpe="sentencepiece",
            sentencepiece_model=SPM_PATH + ".model",
        )
    elif args.arch == "lstm":
        custom_lm = LSTMLanguageModel.from_pretrained(
            os.path.dirname(args.model_path),
            os.path.basename(args.model_path),
            data_name_or_path=DATA_PATH + "/data-bin-ngram"
            if args.add_itag
            else DATA_PATH + "/data-bin",
            bpe="sentencepiece",
            sentencepiece_model=SPM_PATH + ".model",
        )
    else:
        raise NotImplementedError()

    custom_lm.to(device).eval()
    loss_fct = CrossEntropyLoss(ignore_index=-1, reduce=False)
    surprisal4pieces_list = []

    for article, pieces in article2piece.items():
        target_spans = []
        texts = []
        if args.add_bos:
            for context, target in pieces:
                context = context.strip()
                target = target.strip()
                if args.add_itag:
                    context = "</s> <i> " + context
                else:
                    context = "</s> " + context
                target_span = (
                    len(context.split()),
                    len(context.split()) + len(target.split()) - 1,
                )
                target_spans.append(target_span)
                text = context + " " + target
                assert (
                    " ".join(text.split()[target_span[0] : target_span[1] + 1])
                    == target
                )
                texts.append(text.strip())
        else:
            for context, target in pieces:
                context = context.strip()
                target = target.strip()
                if context:
                    if args.add_itag:
                        context = "</s> <i> " + context
                else:
                    context = "</s>"

                # (srart index, end index). For slicing, use as [span[0]:span[1]+1]
                target_span = (
                    len(context.split()),
                    len(context.split()) + len(target.split()) - 1,
                )
                target_spans.append(target_span)
                text = context + " " + target
                text = text.strip()
                assert (
                    " ".join(text.split()[target_span[0] : target_span[1] + 1])
                    == target
                )
                texts.append(text.strip())

        surprisals, surpribsals4pieces = batched_decode(
            custom_lm, texts, target_spans, loss_fct, device, args
        )
        assert len(surprisals) == len(pieces)
        article2scores[article] = surprisals
        surprisal4pieces_list.extend(surpribsals4pieces)

    json.dump(article2scores, open(args.out_dir + "/scores.json", "w"))
    json.dump(
        {"PPL": np.exp(mean(surprisal4pieces_list))},
        open(args.out_dir + "/eval.txt", "w"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", required=True)
    parser.add_argument("--data-path", "-d", required=True)
    parser.add_argument("--add-bos", "-b", action="store_true")
    parser.add_argument("--add-itag", "-i", action="store_true")
    parser.add_argument(
        "--arch", "-a", default="transformer", choices=["transformer", "gpt", "lstm"]
    )
    parser.add_argument("--out-dir", "-o", required=True)
    parser.add_argument("--batchsize", type=int, default=-1)
    parser.add_argument("--corpus", default="BCCWJ", choices=["BCCWJ", "dundee"])
    args = parser.parse_args()
    main(args)
