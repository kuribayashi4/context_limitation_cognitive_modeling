import argparse
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer, AutoModelForCausalLM


def concat_bos(tensor, bos):
    return torch.cat([torch.tensor([bos]), tensor])


@torch.no_grad()
def batched_decode(gpt2_model, text_ids_list, target_spans, loss_fct, device, args):

    random.seed(0)
    pad_id = -100
    batchsize = args.batchsize

    assert len(text_ids_list) == len(target_spans)
    ids_target_spans = [
        (t_s[0], t_s[1], i) for i, t_s in enumerate(zip(text_ids_list, target_spans))
    ]  # [(ids, span, id)]
    ids_target_spans = sorted(ids_target_spans, key=lambda x: len(x[0]))

    sorted_text_ids = list(map(lambda x: x[0], ids_target_spans))
    sorted_target_spans = list(map(lambda x: x[1], ids_target_spans))
    sorted_indices = list(map(lambda x: x[2], ids_target_spans))

    sorted_input_ids = [ids[:-1] for ids in sorted_text_ids]
    sorted_gold_ids = [
        ids[1:] for ids, span in zip(sorted_text_ids, sorted_target_spans)
    ]
    assert len(sorted_input_ids) == len(sorted_gold_ids)

    surprisals4pieces = []
    surprisals4bunsetsu = [0] * len(sorted_indices)
    for i in range(math.ceil(len(text_ids_list) / batchsize)):
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
        input_mask = (padded_input_ids > -1).int()
        padded_input_ids = torch.where(padded_input_ids == pad_id, 0, padded_input_ids)
        padded_gold_ids = torch.where(padded_gold_ids == pad_id, 0, padded_gold_ids)
        assert (padded_input_ids > -1).all()
        assert len(padded_input_ids) == len(padded_gold_ids)

        # SHAPE: (batchsize, max_length, vocab_size)
        results = gpt2_model(
            input_ids=padded_input_ids.to(device), attention_mask=input_mask.to(device)
        )[0]

        for result, gold_ids, span, bunsetsu_id in zip(
            results, padded_gold_ids, batched_target_spans, batched_ids
        ):
            assert (gold_ids < result.shape[1]).all()
            assert len(result) == len(gold_ids)

        batched_positional_scores = loss_fct(
            results.transpose(1, 2), padded_gold_ids.to(device)
        )

        for positional_scores, span, bunsetsu_id in zip(
            batched_positional_scores, batched_target_spans, batched_ids
        ):
            surprisals = (
                positional_scores[span[0] - 1 : span[1]].cpu().detach().clone().tolist()
            )
            surprisals4pieces.extend(surprisals)

            bunsetsu_surprisal = sum(surprisals)
            assert not bunsetsu_surprisal < 0
            surprisals4bunsetsu[bunsetsu_id] = bunsetsu_surprisal

    return surprisals4bunsetsu, surprisals4pieces


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    article2scores = defaultdict(lambda: defaultdict(list))
    article2piece = json.load(open(args.data_path))
    print(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name.replace("_", "-"))
    gpt2_model = AutoModelForCausalLM.from_pretrained(args.model_name.replace("_", "-"))
    gpt2_model.to(device).eval()
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    surprisal4pieces_list = []

    for article, pieces in article2piece.items():
        print(article)
        target_spans: List[Tuple[int]] = []
        text_ids_list = []  # list of tensor(input_ids)
        for context, target in pieces:
            if context.strip():
                context: str = "".join(context.split()).replace(
                    "▁", " "
                )  # with whitespace
                target: str = "".join(target.split()).replace(
                    "▁", " "
                )  # with whitespace
                text = context + target
                # has_context_no_space = 0
            else:
                context = "<|endoftext|>"
                target: str = "".join(target.split()).replace(
                    "▁", " "
                )  # no whitespace (first token in sent.)
                text = context + target

            encoded_context: Dict = tokenizer(context, return_tensors="pt")
            encoded_target: Dict = tokenizer(target, return_tensors="pt")
            encoded_text: Dict = tokenizer(text, return_tensors="pt")
            start: int = len(encoded_context["input_ids"][0])
            target_len: int = len(encoded_target["input_ids"][0])

            target_span: Tuple[int] = (start, start + target_len - 1)
            target_spans.append(target_span)
            target_span_text = tokenizer.decode(
                encoded_text["input_ids"][0][target_span[0] : target_span[1] + 1]
            )
            assert (
                target_span_text == target or " " + target_span_text == target
            )  # gpt2 tokenize -> detokenize removes beginning space before punctuation
            text_ids = encoded_text["input_ids"][0]
            text_ids_list.append(text_ids)

        surprisals, surpribsals4pieces = batched_decode(
            gpt2_model, text_ids_list, target_spans, loss_fct, device, args
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
    parser.add_argument(
        "--model-name",
        "-m",
        required=True,
        choices=["gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl"],
    )
    parser.add_argument("--data-path", "-d", required=True)
    parser.add_argument("--out-dir", "-o", required=True)
    parser.add_argument("--batchsize", type=int, default=-1)
    args = parser.parse_args()
    main(args)
