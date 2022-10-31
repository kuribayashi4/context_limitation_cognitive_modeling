import glob
import json
import math
import os
import re
import logging
import sentencepiece as spm

from logging import getLogger
from collections import defaultdict
from dataclasses import asdict
from itertools import groupby
from statistics import geometric_mean, mean
from typing import List
from pydantic.dataclasses import dataclass
from pydantic import Field

logging.basicConfig(level=logging.DEBUG)
logger = getLogger(__name__)

vocab2freq = json.load(open("models/unigram_en.json"))
sp = spm.SentencePieceProcessor()
sp.Load("models/spm_en/en_wiki.model")

column2key = {
    0: "surface",
    1: "article",
    2: "screenN",
    3: "lineN",
    4: "segmentN",
    5: "tokenN_in_screen",
    6: "characterN_in_line",
    7: "length",
    8: "length_wo_punct",
    9: "punct_code",
    10: "open_punct_code",
    11: "close_punct_code",
    12: "tokenN",
    13: "freq_local",
}


@dataclass
class Token:
    surface: str = ""
    article: int = 0
    screenN: int = 0
    lineN: int = 0
    segmentN: int = 0
    tokenN_in_screen: int = 0
    characterN_in_line: int = 0
    length: int = 0
    length_wo_punct: int = 0
    punct_code: int = 0
    open_punct_code: int = 0
    close_punct_code: int = 0
    tokenN: int = 0
    freq_local: float = 0
    has_num: bool = False
    has_punct: bool = False
    log_gmean_freq: float = 0
    tokenN_in_sent: int = 0
    is_first: bool = False

    is_last: bool = False
    is_second_last: bool = False
    has_num_prev_1: bool = False
    has_punct_prev_1: bool = False


content_pos = [
    "NN",
    "NNP",
    "NNPS",
    "NNS",
]


dep_rel = [
    "nmod",
    "case",
    "det",
    "nsubj",
    "amod",
    "mark",
    "advmod",
    "dobj",
    "conj",
    "aux",
    "cc",
    "compound",
    "acl",
    "cop",
    "advcl",
    "ccomp",
    "xcomp",
    "name",
    "neg",
    "auxpass",
    "parataxis",
    "nummod",
    "nsubjpass",
    "appos",
    "expl",
    "mwe",
    "discourse",
    "csubj",
]


@dataclass
class Leaf:
    wnum: int = 0
    id_in_sent: int = 0
    sent_id: int = 0
    pos: str = ""
    head: int = 0
    dep_rel: str = ""
    child: List = Field(default_factory=list)  # [id, dep_rel]
    preceding_syn_dists: List = Field(default_factory=list)
    preceding_syn_dists_disc: List = Field(default_factory=list)
    anti_locality: int = 0
    avg_locality: float = 0
    min_locality: float = 0
    max_locality: float = 0
    avg_locality_disc: float = 0
    min_locality_disc: float = 0
    max_locality_disc: float = 0
    nmod_dist: int = 0
    nmod_dist_disc: int = 0
    punct_dist: int = 0
    punct_dist_disc: int = 0
    case_dist: int = 0
    case_dist_disc: int = 0
    det_dist: int = 0
    det_dist_disc: int = 0
    nsubj_dist: int = 0
    nsubj_dist_disc: int = 0
    amod_dist: int = 0
    amod_dist_disc: int = 0
    mark_dist: int = 0
    mark_dist_disc: int = 0
    advmod_dist: int = 0
    advmod_dist_disc: int = 0
    root_dist: int = 0
    root_dist_disc: int = 0
    dobj_dist: int = 0
    dobj_dist_disc: int = 0
    conj_dist: int = 0
    conj_dist_disc: int = 0
    aux_dist: int = 0
    aux_dist_disc: int = 0
    cc_dist: int = 0
    cc_dist_disc: int = 0
    compound_dist: int = 0
    compound_dist_disc: int = 0
    acl_dist: int = 0
    acl_dist_disc: int = 0
    cop_dist: int = 0
    cop_dist_disc: int = 0
    advcl_dist: int = 0
    advcl_dist_disc: int = 0
    ccomp_dist: int = 0
    ccomp_dist_disc: int = 0
    xcomp_dist: int = 0
    xcomp_dist_disc: int = 0
    name_dist: int = 0
    name_dist_disc: int = 0
    neg_dist: int = 0
    neg_dist_disc: int = 0
    auxpass_dist: int = 0
    auxpass_dist_disc: int = 0
    parataxis_dist: int = 0
    parataxis_dist_disc: int = 0
    nummod_dist: int = 0
    nummod_dist_disc: int = 0
    nsubjpass_dist: int = 0
    nsubjpass_dist_disc: int = 0
    appos_dist: int = 0
    appos_dist_disc: int = 0
    expl_dist: int = 0
    expl_dist_disc: int = 0
    mwe_dist: int = 0
    mwe_dist_disc: int = 0
    discourse_dist: int = 0
    discourse_dist_disc: int = 0
    csubj_dist: int = 0
    csubj_dist_disc: int = 0

    def calc_content_dist(self, start, end, sent_leaves):
        assert start < end
        return len(
            [
                i
                for i in range(start, end)
                if i in sent_leaves and sent_leaves[i].pos in content_pos
            ]
        )

    def calc_dist_dep_rel(self, dep_type: str, sent_leaves):
        dists = [
            self.id_in_sent - c[0]
            for c in self.child
            if c[1] == dep_type and self.id_in_sent > c[0]
        ]
        if dists:
            dist = mean(dists)
        else:
            dist = 0
        dists_disc = [
            self.calc_content_dist(c[0], self.id_in_sent, sent_leaves)
            for c in self.child
            if c[1] == dep_type and self.id_in_sent > c[0]
        ]
        if dists_disc:
            dist_disc = mean(dists_disc)
        else:
            dist_disc = 0
        return dist, dist_disc

    def calc_preceding_syn_dists(self, sent_leaves):

        if self.id_in_sent > self.head and self.head > 0:
            self.preceding_syn_dists.append(self.id_in_sent - self.head)
        self.preceding_syn_dists.extend(
            [self.id_in_sent - c[0] for c in self.child if self.id_in_sent > c[0]]
        )
        assert not len(self.preceding_syn_dists) or min(self.preceding_syn_dists) > 0

        # only considering content words
        if self.id_in_sent > self.head and self.head > 0:
            self.preceding_syn_dists_disc.append(
                self.calc_content_dist(self.head, self.id_in_sent, sent_leaves)
            )
        for c in self.child:
            if self.id_in_sent > c[0]:
                self.preceding_syn_dists_disc.append(
                    self.calc_content_dist(c[0], self.id_in_sent, sent_leaves)
                )
        assert (
            not len(self.preceding_syn_dists_disc)
            or min(self.preceding_syn_dists_disc) > -1
        )

        self.nmod_dist, self.nmod_dist_disc = self.calc_dist_dep_rel(
            "nmod", sent_leaves
        )
        self.case_dist, self.case_dist_disc = self.calc_dist_dep_rel(
            "case", sent_leaves
        )
        self.det_dist, self.det_dist_disc = self.calc_dist_dep_rel("det", sent_leaves)
        self.nsubj_dist, self.nsubj_dist_disc = self.calc_dist_dep_rel(
            "nsubj", sent_leaves
        )
        self.amod_dist, self.amod_dist_disc = self.calc_dist_dep_rel(
            "amod", sent_leaves
        )
        self.mark_dist, self.mark_dist_disc = self.calc_dist_dep_rel(
            "mark", sent_leaves
        )
        self.advmod_dist, self.advmod_dist_disc = self.calc_dist_dep_rel(
            "advmod", sent_leaves
        )
        self.dobj_dist, self.dobj_dist_disc = self.calc_dist_dep_rel(
            "dobj", sent_leaves
        )
        self.conj_dist, self.conj_dist_disc = self.calc_dist_dep_rel(
            "conj", sent_leaves
        )
        self.aux_dist, self.aux_dist_disc = self.calc_dist_dep_rel("aux", sent_leaves)
        self.cc_dist, self.cc_dist_disc = self.calc_dist_dep_rel("cc", sent_leaves)
        self.compound_dist, self.compound_dist_disc = self.calc_dist_dep_rel(
            "compound", sent_leaves
        )
        self.acl_dist, self.acl_dist_disc = self.calc_dist_dep_rel("acl", sent_leaves)
        self.cop_dist, self.cop_dist_disc = self.calc_dist_dep_rel("cop", sent_leaves)
        self.advcl_dist, self.advcl_dist_disc = self.calc_dist_dep_rel(
            "advcl", sent_leaves
        )
        self.ccomp_dist, self.ccomp_dist_disc = self.calc_dist_dep_rel(
            "ccomp", sent_leaves
        )
        self.xcomp_dist, self.xcomp_dist_disc = self.calc_dist_dep_rel(
            "xcomp", sent_leaves
        )
        self.name_dist, self.name_dist_disc = self.calc_dist_dep_rel(
            "name", sent_leaves
        )
        self.neg_dist, self.neg_dist_disc = self.calc_dist_dep_rel("neg", sent_leaves)
        self.auxpass_dist, self.auxpass_dist_disc = self.calc_dist_dep_rel(
            "auxpass", sent_leaves
        )
        self.parataxis_dist, self.parataxis_dist_disc = self.calc_dist_dep_rel(
            "parataxis", sent_leaves
        )
        self.nsubjpass_dist, self.nsubjpass_dist_disc = self.calc_dist_dep_rel(
            "nsubjpass", sent_leaves
        )
        self.appos_dist, self.appos_dist_disc = self.calc_dist_dep_rel(
            "appos", sent_leaves
        )
        self.expl_dist, self.expl_dist_disc = self.calc_dist_dep_rel(
            "expl", sent_leaves
        )
        self.mwe_dist, self.mwe_dist_disc = self.calc_dist_dep_rel("mwe", sent_leaves)
        self.discourse_dist, self.discourse_dist_disc = self.calc_dist_dep_rel(
            "discourse", sent_leaves
        )
        self.csubj_dist, self.csubj_dist_disc = self.calc_dist_dep_rel(
            "csubj", sent_leaves
        )

    def calc_anti_locality(self):
        self.anti_locality = len(self.preceding_syn_dists)

    def calc_locality(self):
        if len(self.preceding_syn_dists) > 0:
            self.avg_locality = mean(self.preceding_syn_dists)
            self.min_locality = min(self.preceding_syn_dists)
            self.max_locality = max(self.preceding_syn_dists)
            self.avg_locality_disc = mean(self.preceding_syn_dists_disc)
            self.min_locality_disc = min(self.preceding_syn_dists_disc)
            self.max_locality_disc = max(self.preceding_syn_dists_disc)


@dataclass
class DataPoint(Token, Leaf):
    subj_id: str = ""
    time: int = 0
    logtime: float = 0
    length_prev_1: int = 0
    log_gmean_freq_prev_1: float = 0


def calc_freq(pieces: str):
    gmean = geometric_mean(
        [vocab2freq[t] + 0.001 if t in vocab2freq else 0.001 for t in pieces.split()]
    )
    log_gmean = math.log(gmean)
    return log_gmean


def load_tokens(files):
    article2tokens = defaultdict(list)
    for file in files:
        with open(file) as f:
            text_id = int(os.path.basename(file)[2:4])
            lines = f.readlines()
            tokenN = 1
            for i, line in enumerate(lines):
                line = line.strip()
                line = re.sub("\s+", "\t", line)
                info_from_line_dict = {
                    item[1]: col
                    for col, item in zip(
                        line.split("\t"), sorted(column2key.items(), key=lambda x: x[0])
                    )
                }
                info_from_line_dict["surface"] = " ".join(
                    sp.EncodeAsPieces(info_from_line_dict["surface"])
                )
                token = Token(
                    **info_from_line_dict,
                    has_num=bool(re.findall(r"[0-9]", info_from_line_dict["surface"])),
                    has_punct=int(info_from_line_dict["punct_code"]) > 0,
                    is_first=info_from_line_dict["tokenN_in_screen"] == "1",
                    log_gmean_freq=calc_freq(info_from_line_dict["surface"]),
                    tokenN_in_sent=tokenN,
                )
                article2tokens[text_id].append(token)
                if (
                    token.surface.endswith(".")
                    or token.surface.endswith("!")
                    or token.surface.endswith("?")
                ):
                    tokenN = 1
                else:
                    tokenN += 1

        tokens = article2tokens[text_id]
        for i, token in reversed(list(enumerate(tokens))):
            if i == len(article2tokens[text_id]) - 1:
                article2tokens[text_id][i].is_last = True
                continue
            elif int(tokens[i + 1].tokenN_in_screen) < token.tokenN_in_screen:
                article2tokens[text_id][i].is_last = True
            elif tokens[i + 1].is_last:
                article2tokens[text_id][i].is_second_last = True
            if i > 0:
                if tokens[i - 1].has_num:
                    article2tokens[text_id][i].has_num_prev_1 = True
                if tokens[i - 1].has_punct:
                    article2tokens[text_id][i].has_punct_prev_1 = True

    return article2tokens


def load_treebank(treebank_files):
    article2leaves = defaultdict(dict)
    for treebank in treebank_files:
        article_id = int(os.path.basename(treebank)[2:4])
        with open(treebank) as f:
            for sent_id, sent_lines in groupby(f, lambda x: x.split("\t")[2]):
                sent_leaves = {}
                sent_lines = list(sent_lines)
                for line in sent_lines:
                    info = line.strip().split("\t")
                    if info[-1] == "punct":
                        continue
                    leaf = Leaf(
                        wnum=info[1],
                        id_in_sent=info[3],
                        sent_id=info[2],
                        pos=info[4],
                        head=info[5],
                        dep_rel=info[6],
                    )
                    sent_leaves[int(info[3])] = leaf

                # collecting children
                for _, sent_leaf in sent_leaves.items():
                    if sent_leaf.head > 0:
                        if sent_leaves.get(sent_leaf.head):
                            sent_leaves[sent_leaf.head].child.append(
                                (sent_leaf.id_in_sent, sent_leaf.dep_rel)
                            )

                # calculating syntactic scores
                for _, sent_leaf in sent_leaves.items():
                    sent_leaf.calc_preceding_syn_dists(sent_leaves)
                    sent_leaf.calc_anti_locality()
                    sent_leaf.calc_locality()
                article2leaves[article_id].update(
                    {leaf.wnum: leaf for leaf in sent_leaves.values()}
                )
    return article2leaves


def load_durations(files, article2tokens):
    subj2first_duration = defaultdict(lambda: defaultdict(dict))
    for file in files:
        with open(file) as f:
            subj_id = os.path.basename(file)[0:2]
            text_id = int(os.path.basename(file)[2:4])
            prev_wnum = None
            for line in f:
                line = line.strip()
                line = re.sub("\s+", "\t", line)
                if line.split()[0] == "WORD":
                    continue
                if line.split()[6] == "-99":
                    continue
                if line.split()[6] == "0":
                    continue
                info = line.split("\t")
                wnum = int(info[6])
                duration = int(info[7])
                if wnum == prev_wnum:
                    subj2first_duration[subj_id][text_id][wnum] += duration
                elif subj2first_duration[subj_id][text_id].get(wnum):
                    continue
                else:
                    subj2first_duration[subj_id][text_id][wnum] = duration
                prev_wnum = wnum
    return subj2first_duration


def merge_token_duration(article2tokens, subj2first_duration, article2leaves):
    data_points = []
    for subj_id, article2duration in subj2first_duration.items():
        for text_id, wnum2duration in sorted(
            article2duration.items(), key=lambda x: x[0]
        ):
            tokens = article2tokens[text_id]
            tree_info = article2leaves[text_id]

            length_prev_1 = 4.86941  # pre-computed avg. value
            log_gmean_freq_prev_1 = 11.0747
            for i, token in enumerate(tokens):
                duration = wnum2duration.get(i + 1)
                leaf = tree_info.get(i + 1)
                if duration:
                    data_point = DataPoint(
                        **asdict(token),
                        **asdict(leaf) if leaf else {},
                        subj_id=subj_id,
                        time=duration,
                        logtime=math.log10(duration),
                        length_prev_1=length_prev_1,
                        log_gmean_freq_prev_1=log_gmean_freq_prev_1,
                    )
                else:
                    data_point = DataPoint(
                        **asdict(token),
                        **asdict(leaf) if leaf else {},
                        subj_id=subj_id,
                        time=0,
                        logtime="-Infinity",
                        length_prev_1=length_prev_1,
                        log_gmean_freq_prev_1=log_gmean_freq_prev_1,
                    )
                data_points.append(asdict(data_point))
                length_prev_1 = token.length
                log_gmean_freq_prev_1 = token.log_gmean_freq
    return data_points


def main():
    files = glob.glob("data/DC/dundee_corpus_utf8/tx*")
    article2tokens = load_tokens(files)
    logger.info("loaded texts")

    article2pieces = {
        article_id: [t.surface for t in tokens]
        for article_id, tokens in article2tokens.items()
    }
    json.dump(article2pieces, open("data/DC/pieces.json", "w"))

    duration_files = glob.glob("data/DC/dundee_corpus_utf8/*ma1p.dat")
    subj2durations = load_durations(duration_files, article2tokens)
    logger.info("loaded gaze durations")

    treebanks = glob.glob("data/DC/treebank/*.modified")
    article2leaves = load_treebank(treebanks)
    logger.info("loaded treebanks")

    data_points = merge_token_duration(article2tokens, subj2durations, article2leaves)
    logger.info("merged annotations")
    logger.info(data_points[:10])
    json.dump(data_points, open("data/DC/all.txt.annotation", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
