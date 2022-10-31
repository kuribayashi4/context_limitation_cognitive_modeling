import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class Experiment:
    arch: str
    context_l: str
    context_func: str
    seed: str


def parse_path(path):
    info = path.split("/")
    if "gpt2" in info[-2]:
        arch, context_l, *context_func = info[-2].split("-")
        context_func = "-".join(context_func)
        experiment = Experiment(
            **{
                "arch": arch.replace("arch_", ""),
                "context_l": context_l.split("_")[-1],
                "context_func": "_".join(context_func.split("_")[1:]),
                "seed": "-",
            }
        )
    else:
        arch, context_l, *context_func = info[-3].split("-")
        seed = info[-2].replace("seed-", "")
        context_func = "-".join(context_func)
        experiment = Experiment(
            **{
                "arch": arch.split("_")[-1],
                "context_l": context_l.split("_")[1],
                "context_func": "_".join(context_func.split("_")[1:]),
                "seed": seed,
            }
        )
    return experiment


def main():
    lls = glob.glob(f"{args.dir}/**/{args.file}", recursive=True)
    ppls = glob.glob(f"{args.dir}/**/eval.txt", recursive=True)

    dir2scores = defaultdict(dict)
    dir2experiment = {}

    for ll in lls:
        dir = os.path.dirname(ll)
        experiment = parse_path(ll)
        dir2experiment[dir] = experiment
        with open(ll) as f:
            for line in f:
                dir2scores[dir][line.split(args.separator)[0]] = str(
                    line.split(args.separator)[1].strip()
                )

    if not args.file_only:
        for ppl in ppls:
            dir = os.path.dirname(ppl)
            if dir in dir2experiment:
                scores = json.load(open(ppl))
                for k, v in scores.items():
                    if k == "PPL":
                        dir2scores[dir][k] = str(v)

    flag_printed_head = 0
    assert len(set([len(v) for v in dir2scores.values()])) == 1

    for dir, scores in sorted(dir2scores.items(), key=lambda x: x[0]):
        if not flag_printed_head:
            print(
                ",".join(s[0] for s in sorted(scores.items(), key=lambda x: x[0]))
                + ","
                + ",".join(sorted(asdict(dir2experiment[dir]).keys()))
            )
            flag_printed_head = 1
        print(
            ",".join(s[1] for s in sorted(scores.items(), key=lambda x: x[0])), end=","
        )
        if dir in dir2experiment:
            print(
                ",".join(
                    [
                        v
                        for k, v in sorted(
                            asdict(dir2experiment[dir]).items(), key=lambda x: x[0]
                        )
                    ]
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--file-only", action="store_true")
    parser.add_argument("--separator", default=":")
    args = parser.parse_args()
    main()
