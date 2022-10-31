import glob
import json
import math
import os
import unicodedata
import logging
import MeCab
import mojimoji
import sentencepiece as spm

from logging import getLogger
from collections import defaultdict, deque
from dataclasses import asdict
from itertools import groupby
from statistics import geometric_mean, mean
from typing import Dict, List, Tuple
from pydantic.dataclasses import dataclass
from pydantic import Field

logging.basicConfig(level=logging.DEBUG)
logger = getLogger(__name__)

UNIDIC_PATH = "/opt/local/lib/mecab/dic/unidic"  # should be changed
DEPPARA_PATH = "data/BE/BCCWJ-DepPara"
LUW_PATH = "data/BE/BCCWJ-LUW"
WORD_PATH = "data/BE/word.txt"
MORPH_PATH = "data/BE/morph.tsv"
vocab2freq = json.load(open("models/unigram_ja.json"))
m = MeCab.Tagger(f"-d {UNIDIC_PATH}")
sp = spm.SentencePieceProcessor()
sp.Load("models/spm_ja/japanese_gpt2_unidic.model")

origin_key_list = [
    "logtime",
    "invtime",
    "segment",
    "time",
    "measure",
    "sample",
    "article",
    "metadata_orig",
    "metadata",
    "sessionN",
    "length",
    "space",
    "subj",
    "setorder",
    "rspan",
    "voc",
    "dependent",
    "articleN",
    "screenN",
    "lineN",
    "segmentN",
    "sample_screen",
    "is_first",
    "is_last",
    "is_second_last",
    "infostatus",
    "definiteness",
    "specificity",
    "animacy",
    "sentience",
    "agentivity",
    "commonness",
    "CLRHST",
    "CLRMST",
    "CLRFUT",
    "CLRHRT",
    "CLRHSL",
    "CLRMSL",
    "CLRFUL",
    "CLRHRL",
    "CLMHST",
    "CLMMST",
    "CLMFUT",
    "CLMHRT",
    "CLMHSL",
    "CLMMSL",
    "CLMFUL",
    "CLMHRL",
    "WLSPSUWA",
    "WLSPSUWB",
    "WLSPSUWC",
    "WLSPSUWD",
    "WLSPLUWA",
    "WLSPLUWB",
    "WLSPLUWC",
    "WLSPLUWD",
    "type_pred",
    "dist_invga",
    "type_ga",
    "dist_invo",
    "type_o",
    "dist_invni",
    "type_ni",
]


content_pos = [
    "NOUN",
]


@dataclass
class Bunsetsu:
    """
    tentative object for calculating locality and anti-locality scores
    """

    id: int
    parent: int
    pos: List[str]
    dep_rel: List[str]

    child: List = Field(default_factory=list)
    avg_locality: float = (
        0  # average of distance between this and its preceding dependents
    )
    min_locality: float = 0  # min of distance between this and its preceding dependents
    max_locality: float = 0  # max of distance between this and its preceding dependents
    avg_locality_disc: float = 0
    min_locality_disc: float = 0
    max_locality_disc: float = 0
    anti_locality: int = 0
    surface_deppara: str = ""
    preceding_syn_dists: List = Field(default_factory=list)
    preceding_syn_dists_disc: List = Field(default_factory=list)
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
    obj_dist: int = 0
    obj_dist_disc: int = 0
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
    obl_dist: int = 0
    obl_dist_disc: int = 0
    dep_dist: int = 0
    dep_dist_disc: int = 0
    fixed_dist: int = 0
    fixed_dist_disc: int = 0

    def calc_content_dist(self, start, end, sent_leaves):
        assert start < end
        return len(
            [
                i
                for i in range(start, end)
                if i < len(sent_leaves) and (set(sent_leaves[i].pos) & set(content_pos))
            ]
        )

    def calc_dist_dep_rel(self, dep_type: str, sent_leaves):
        dists = [
            self.id - c[0] for c in self.child if dep_type in c[1] and self.id > c[0]
        ]
        if dists:
            dist = mean(dists)
        else:
            dist = 0
        dists_disc = [
            self.calc_content_dist(c[0], self.id, sent_leaves)
            for c in self.child
            if dep_type in c[1] and self.id > c[0]
        ]
        if dists_disc:
            dist_disc = mean(dists_disc)
        else:
            dist_disc = 0
        return dist, dist_disc

    def calc_preceding_syn_dists(self, sent_leaves):
        # memo: this function is called multiple times. Be careful not to append too many elements to array-type attributes.

        self.preceding_syn_dists = []
        self.preceding_syn_dists_disc = []
        if self.id > self.parent and self.parent > -1:
            self.preceding_syn_dists.append(self.id - self.parent)
        self.preceding_syn_dists = self.preceding_syn_dists + [
            self.id - c[0] for c in self.child if self.id > c[0]
        ]
        assert not len(self.preceding_syn_dists) or min(self.preceding_syn_dists) > 0

        # only considering content words
        if self.id > self.parent and self.parent > -1:
            self.preceding_syn_dists_disc.append(
                self.calc_content_dist(self.parent, self.id, sent_leaves)
            )
        for c in self.child:
            if self.id > c[0]:
                self.preceding_syn_dists_disc.append(
                    self.calc_content_dist(c[0], self.id, sent_leaves)
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
        self.obj_dist, self.obj_dist_disc = self.calc_dist_dep_rel("obj", sent_leaves)
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
        self.obl_dist, self.obl_dist_disc = self.calc_dist_dep_rel("obl", sent_leaves)
        self.dep_dist, self.dep_dist_disc = self.calc_dist_dep_rel("dep", sent_leaves)
        self.fixed_dist, self.fixed_dist_disc = self.calc_dist_dep_rel(
            "fixed", sent_leaves
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
class DataPoint(Bunsetsu):
    surface: str = ""
    logtime: float = 0
    invtime: float = 0
    segment: str = ""
    time: int = 0
    measure: str = ""
    sample: str = ""
    article: str = ""
    metadata_orig: str = ""
    metadata: str = ""
    sessionN: int = 0
    length: int = 0
    space: int = 0
    subj: str = ""
    setorder: str = ""
    rspan: float = 0
    voc: float = 0
    dependent: int = 0
    articleN: int = 0
    screenN: int = 0
    lineN: int = 0
    segmentN: int = 0
    sample_screen: str = ""
    is_first: bool = False
    is_last: bool = False
    is_second_last: bool = False
    infostatus: str = ""
    definiteness: str = ""
    specificity: str = ""
    animacy: str = ""
    sentience: str = ""
    agentivity: str = ""
    commonness: str = ""
    CLRHST: bool = False
    CLRMST: bool = False
    CLRFUT: bool = False
    CLRHRT: bool = False
    CLRHSL: str = ""
    CLRMSL: str = ""
    CLRFUL: str = ""
    CLRHRL: str = ""
    CLMHST: bool = False
    CLMMST: bool = False
    CLMFUT: bool = False
    CLMHRT: bool = False
    CLMHSL: str = ""
    CLMMSL: str = ""
    CLMFUL: str = ""
    CLMHRL: str = ""
    WLSPSUWA: str = ""
    WLSPSUWB: str = ""
    WLSPSUWC: str = ""
    WLSPSUWD: str = ""
    WLSPLUWA: str = ""
    WLSPLUWB: str = ""
    WLSPLUWC: str = ""
    WLSPLUWD: str = ""
    type_pred: str = ""
    dist_invga: float = 0
    type_ga: str = ""
    dist_invo: float = 0
    type_o: str = ""
    dist_invni: float = 0
    type_ni: str = ""
    bos: bool = False
    eos: bool = False
    b_eos: bool = False
    tokenN: int = 0
    sentN: int = 0
    log_gmean_freq: float = 0
    has_num: bool = False
    has_punct: bool = False

    length_prev_1: int = 0
    log_gmean_freq_prev_1: float = 0
    pos_luw: List[str] = Field(default_factory=list)


def load_bccwj_deppara(DEP_PARA_PATH):
    doc_ids = [
        os.path.basename(id[:-11])
        for id in glob.glob(os.path.join(DEP_PARA_PATH, "*.ud"))
    ]
    article2bunsetsu: Dict[str, List[Bunsetsu]] = {}

    for doc_id in doc_ids:
        bunsetsu_list_doc = []
        with open(f"data/BE/BCCWJ-DepPara/{doc_id}.cabocha.ud") as f:
            for is_eos, lines in groupby(f, key=lambda x: x.strip() == "EOS"):
                if not is_eos:
                    lines = list(lines)
                    bunsetsu_list: List[Bunsetsu] = []
                    current_bunsetsu = None
                    for line in lines:
                        if line[0:2] == "#!":
                            continue
                        elif line[0:2] == "* ":
                            if current_bunsetsu:
                                bunsetsu_list.append(current_bunsetsu)
                            _, id, parent, _, _, pos, dep_rel, *_ = line.split()
                            pos = pos.split(",")
                            dep_rel = dep_rel.split(",")
                            current_bunsetsu = Bunsetsu(
                                id=int(id),
                                parent=int(parent.rstrip("DZBFO")),
                                pos=pos,
                                dep_rel=dep_rel,
                            )
                        else:
                            if current_bunsetsu and line.split("\t")[0] != "\u3000":
                                current_bunsetsu.surface_deppara += line.split("\t")[0]

                    if current_bunsetsu and current_bunsetsu.surface_deppara:
                        bunsetsu_list.append(current_bunsetsu)

                    for b in bunsetsu_list:
                        if b.parent > -1:
                            bunsetsu_list[b.parent].child.append((b.id, b.dep_rel))

                    if bunsetsu_list:
                        for b in bunsetsu_list:
                            b.calc_preceding_syn_dists(bunsetsu_list)
                            b.calc_anti_locality()
                            b.calc_locality()

                    bunsetsu_list_doc.extend(bunsetsu_list)
        article2bunsetsu[doc_id] = bunsetsu_list_doc
    return article2bunsetsu


def load_surfaces(WORD_PATH, MORPH_PATH) -> None:
    article2pieces: Dict[str, List[str]] = defaultdict(list)
    article2bos: Dict[str, List[str]] = defaultdict(list)

    with open(WORD_PATH) as f, open(MORPH_PATH) as fm:
        morphs = deque([line.split()[0] for line in fm if not line.strip() == "EOS"])
        for id, lines in groupby(f, key=lambda x: x.split("\t")[7]):
            if id != "article":
                lines = list(lines)
                article_pieced = []
                bos_list = []
                for bunsetsu, info in groupby(lines, key=lambda x: x.split("\t")[5]):
                    bunsetsu = unicodedata.normalize(
                        "NFKC", mojimoji.han_to_zen(bunsetsu)
                    )
                    info = list(info)
                    bos = info[-1].split("\t")[-1].strip()
                    bunsetsu_pieced = []
                    while bunsetsu:
                        assert morphs[0] == bunsetsu[: len(morphs[0])]
                        bunsetsu_pieced.append(morphs.popleft())
                        bunsetsu = bunsetsu[len(bunsetsu_pieced[-1]) :]
                    pieces = sp.EncodeAsPieces(" ".join(bunsetsu_pieced))
                    pieces = " ".join(pieces)
                    article_pieced.append(pieces)
                    bos_list.append(bos)
                article2pieces[id] = article_pieced
                article2bos[id] = bos_list
    return article2pieces, article2bos


def load_luws(LUW_PATH):
    files = glob.glob(f"{LUW_PATH}/*.tsv")
    article2luws = {}
    for file in files:
        article_id = os.path.basename(file.split(".")[0])
        with open(file) as f:
            luw_lines = f.readlines()
            article2luws[article_id] = [tuple(l.strip().split("\t")) for l in luw_lines]
    return article2luws


def calc_freq(pieces: str):
    toks = pieces.split()
    gmean = geometric_mean(
        [vocab2freq[t] + 0.001 if t in vocab2freq else 0.001 for t in toks]
    )
    log_gmean = math.log(gmean)
    return log_gmean


def has_num(pieces: str):
    bunsetsu = pieces.replace(" ", "").replace("▁", "")
    if len(
        [p for p in m.parse(bunsetsu).split("\n") if "数" in "".join(p.split("\t")[4:])]
    ):
        return True
    else:
        return False


def has_punct(pieces: str):
    bunsetsu = pieces.replace(" ", "").replace("▁", "")
    if len(
        [p for p in m.parse(bunsetsu).split("\n") if "記号" in "".join(p.split("\t")[4:])]
    ):
        return True
    else:
        return False


def annotate(file, article2pieces, article2bos, article2bunsetsu, article2luws):
    bos2id: Dict[str, int] = {"B": True, "I": False}
    outputs: List[str] = []
    with open(file) as f:
        tokenN = 1
        for id, lines in groupby(f, key=lambda x: x.split(",")[6]):

            # each article
            sentN = 0
            lines = list(lines)
            length_prev_1 = 5.17833  # pre-computed avg. value
            log_gmean_freq_prev_1 = 13.9232  # pre-computed avg. value
            if id == "article":
                pass
            else:
                assert len(lines) == len(article2pieces[id])
                assert len(lines) == len(article2bos[id])
                assert len(lines) == len(article2bunsetsu[id])
                luws: List[Tuple[str]] = article2luws["_".join(id.split("_")[2:])]

                # then annotate the info
                for i, info in enumerate(
                    zip(
                        lines,
                        article2pieces[id],
                        article2bos[id],
                        article2bunsetsu[id],
                    )
                ):
                    line, piece, bos, bunsetsu = info
                    n_pieces: int = len(piece.split())
                    bunsetsu_luws = luws[:n_pieces]
                    luws = luws[n_pieces:]
                    assert " ".join([s for s, p in bunsetsu_luws]) == piece

                    eos = False
                    b_eos = False  # whether the bunsets is before eos
                    if i < len(lines) - 1:
                        if article2bos[id][i + 1] == "B":
                            eos = True
                    if i < len(lines) - 2:
                        if article2bos[id][i + 2] == "B" and not eos:
                            b_eos = True
                    if bos == "B":
                        tokenN = 1
                        sentN += 1

                    info_from_line_list = line.strip().split(",")
                    assert len(origin_key_list) == len(info_from_line_list)
                    info_from_line_dict = {
                        k: v for k, v in zip(origin_key_list, info_from_line_list)
                    }
                    log_gmean_freq = calc_freq(piece)

                    assert bunsetsu.avg_locality <= bunsetsu.id
                    assert not bunsetsu.child or bunsetsu.anti_locality > 0

                    data_ponit = DataPoint(
                        surface=piece,
                        bos=bos2id[bos],
                        eos=eos,
                        b_eos=b_eos,
                        tokenN=tokenN,
                        sentN=sentN,
                        has_num=has_num(piece),
                        has_punct=has_punct(piece),
                        log_gmean_freq=log_gmean_freq,
                        length_prev_1=length_prev_1,
                        log_gmean_freq_prev_1=log_gmean_freq_prev_1,
                        pos_luw=[p for s, p in bunsetsu_luws],
                        **info_from_line_dict,
                        **asdict(bunsetsu),
                    )
                    outputs.append(asdict(data_ponit))
                    tokenN += 1
                    length_prev_1 = info_from_line_dict["length"]
                    log_gmean_freq_prev_1 = log_gmean_freq
    return outputs


def main():
    article2pieces, article2bos = load_surfaces(WORD_PATH, MORPH_PATH)
    json.dump(article2pieces, open("data/BE/pieces.json", "w"), ensure_ascii=False)

    article2bunsetsu = load_bccwj_deppara(DEPPARA_PATH)
    article2luws = load_luws(LUW_PATH)

    file = "data/BE/fpt-log.csv"
    outputs = annotate(
        file, article2pieces, article2bos, article2bunsetsu, article2luws
    )
    logger.info(outputs[:10])
    json.dump(outputs, open(f"{file}.annotation", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
