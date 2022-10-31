import argparse
import os
import json
import glob
import tqdm
import sys
from statistics import mean


def concat_results(scores, article_list, prev: int = -1):
    if prev > 0:
        results = []
        for a in article_list:
            avg_score = mean(scores[a])
            results.extend([avg_score] * prev + scores[a][: -1 * prev])
        return results
    else:
        return [s for a in article_list for s in scores[a]]


def add_prev_feature(df, key, i):
    df[f"{key}_prev_{i}"] = [0] * i + df[key].tolist()[: -1 * i]


def main():
    if args.corpus == "BCCWJ":
        with open("data/BE/article_order.txt") as f:
            article_list = [a.strip() for a in f]

    elif args.corpus == "dundee":
        with open("data/DC/article_order.txt") as f:
            article_list = [a.strip() for a in f]
    else:
        raise NotImplementedError

    if args.dir:
        input_files = glob.glob(args.dir + "/**/scores.json", recursive=True)
    elif args.input:
        input_files = [args.input]
    else:
        print("Please set input file/dir.")
        sys.exit(1)

    for input_file in tqdm.tqdm(input_files):
        print(input_file)
        dir = os.path.dirname(input_file)
        article2scores = json.load(open(input_file))

        # test
        if args.corpus == "BCCWJ":
            assert "PN1c_00001_A_1" in article2scores
        elif args.corpus == "dundee":
            assert "1" in article2scores
        else:
            raise NotImplementedError

        with open(os.path.join(dir, "scores.csv"), "w") as f:
            header = "\t".join(
                [
                    "surprisals_sum" if prev == 0 else f"surprisals_sum_prev_{prev}"
                    for prev in range(0, 4)
                ]
            )
            f.write(header + "\n")
            scores = {}
            scores["surprisals_sum"] = concat_results(article2scores, article_list)
            for i in range(1, 4):
                scores["surprisals_sum_prev_" + str(i)] = concat_results(
                    article2scores, article_list, prev=i
                )

            f.write(
                "\n".join(
                    [
                        "\t".join([str(scores[h][i]) for h in header.split("\t")])
                        for i in range(len(scores["surprisals_sum"]))
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--dir", default="")
    parser.add_argument("--corpus", default="BCCWJ", choices=["BCCWJ", "dundee"])
    args = parser.parse_args()
    main()
