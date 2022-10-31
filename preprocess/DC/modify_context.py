import csv
import json
import random
import argparse
from functools import partial
from itertools import groupby
from typing import Dict, List, Tuple


def extract_subwords(bunsetsu_list):
    return " ".join(s for b in bunsetsu_list for s in b["surface"].split())


def _delete(bunsetsu_list: List[dict], ngram):
    if ngram == 1000:
        return [
            (extract_subwords(bunsetsu_list[:i]), extract_subwords([b]))
            for i, b in enumerate(bunsetsu_list)
        ]

    return [
        (
            extract_subwords(bunsetsu_list[max(0, i - ngram + 1) : i]),
            extract_subwords([b]),
        )
        for i, b in enumerate(bunsetsu_list)
    ]


def _delete_lossy(bunsetsu_list: List[dict], ngram, slope):
    def _probablistic_delete(bunsetsu_list, slope):
        return [
            bunsetsu
            for i, bunsetsu in enumerate(bunsetsu_list, start=1)
            if random.choices(
                [0, 1],
                weights=[
                    min(1, i * slope),
                    max(0, 1 - i * slope),
                ],
            )[0]
        ]

    out = []
    for i, b in enumerate(bunsetsu_list):
        context = bunsetsu_list[:i]
        far_context = context[: -ngram + 1]
        far_context = _probablistic_delete(far_context[::-1], slope)[::-1]
        close_context = context[-ngram + 1 :]
        out.append(
            (
                extract_subwords(far_context) + " " + extract_subwords(close_context),
                extract_subwords([b]),
            )
        )
    return out


def attach_context(lines, modify_context, ngram=1000):
    bunsetsu_list: List[Tuple[str, str]] = []
    for sent_id, sent in groupby(lines, key=lambda x: x["sent_id"]):
        sent = list(sent)
        bunsetsu_list.extend(modify_context(sent, ngram))
    return bunsetsu_list


def main(args):
    if args.context_func == "delete":
        modify_context = _delete
        filename = f"ngram_{args.n}-contextfunc_{args.context_func}"
    elif args.context_func == "lossy":
        modify_context = partial(_delete_lossy, slope=args.lossy_slope)
        filename = f"ngram_{args.n}-contextfunc_{args.context_func}-{args.lossy_slope}"
    article2pieces_context = {}
    with open("data/DC/all.txt.annotation.filtered.csv") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for article_id, lines in groupby(reader, lambda x: x["article"]):
            if article_id not in article2pieces_context:
                print(article_id)
                lines: List[Dict[str, str]] = list(lines)
                out = attach_context(
                    lines,
                    modify_context=modify_context,
                    ngram=args.n,
                )
                assert len(lines) == len(out)
                article2pieces_context[article_id] = out

    json.dump(
        article2pieces_context,
        open(f"data/DC/{filename}.json", "w"),
        ensure_ascii=False,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--lossy-slope", type=float, default=0.1)
    parser.add_argument(
        "--context-func",
        choices=[
            "delete",
            "lossy",
        ],
        default="delete",
    )
    args = parser.parse_args()

    random.seed(1234)
    main(args)
