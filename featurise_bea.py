import csv
import multiprocessing as mp

from pathlib import Path
from string import punctuation
from typing import Any, Generator

import click
import spacy
import numpy as np
import pandas as pd

from spacy.tokens import Doc
from wordfreq import zipf_frequency


def load_psy_feature_dict(tsv_fpath: Path):
    psy_feature_dict = {}
    pos_map = {"Noun": "NOUN", "Verb": "VERB", "Adjective": "ADJ", "Adverb": "ADV"}
    # converting to spacy tags

    with open(tsv_fpath, "r", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")

        for row in reader:
            # print(row)
            org_pos = row["Dom_PoS_SUBTLEX"]
            if org_pos not in pos_map:
                continue

            converted_pos = pos_map[org_pos]

            identifier = (row["Word"], converted_pos)
            psy_feature_dict[identifier] = {
                "freq_pm": row["Freq_pm"],
                "aoa": row["AoA_Kup_lem"],
            }

    return psy_feature_dict


def load_cefr_j_dict(tsv_fpath: Path) -> dict[tuple[str, str], int]:
    pos_map = {
        "noun": "NOUN",
        "verb": "VERB",
        "adjective": "ADJ",
        "adverb": "ADV",
        "be-verb": "VERB",
        "do-verb": "VERB",
        "have-verb": "VERB",
    }

    cefr_map = {
        "A1": 0,
        "A2": 1,
        "B1": 2,
        "B2": 3,
        "C1": 4,
        "C2": 5
    }

    cefr_j_dict: dict[tuple[str, str], int] = {}
    with open(tsv_fpath, "r", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")

        for row in reader:
            # print(row)
            org_pos = row["pos"]
            if org_pos not in pos_map:
                continue

            converted_pos = pos_map[org_pos]
            identifier = (row["word"], converted_pos)
            cefr_j_dict[identifier] = cefr_map[row["Cefr-j"]]

    return cefr_j_dict


def get_sequence_feature(token_list: list[str]) -> dict[str, Any]:
    sequence_length = len(token_list)

    return {"sequence_length": sequence_length}


def get_token_features(token: str) -> dict[str, Any]:
    character_length = len(token)
    stripped_token = token.strip().strip(punctuation)
    frequency = zipf_frequency(stripped_token, "en")

    first_upper = False
    all_upper = False

    if len(stripped_token) > 0:
        first_upper = stripped_token[0].isupper()
        all_upper = stripped_token.isupper()

    return {
        "character_length": character_length,
        "frequency": frequency,
        "first_upper": first_upper,
        "all_upper": all_upper,
    }


def process_row(
    row: pd.Series,
    nlp,
    psy_feature_dict: dict[tuple[str, str], dict[str, float]],
    cefr_j_dict: dict[tuple[str, str], int],
) -> Generator[dict[str, Any], None, None]:
    token_seq = row["token_seq"]

    doc = nlp(token_seq)
    pos_tags = [t.pos_ for t in doc]
    lemmas = [t.lemma_ for t in doc]

    token_list = token_seq.split("|||")
    labels = row["label_seq"].split("|||")
    cefr = row["cefr"]

    sequence_features = get_sequence_feature(token_list)

    for token, pos, lem, lab in zip(token_list, pos_tags, lemmas, labels):
        # if pos not in ("NOUN", "VERB", "ADJ", "ADV"):
        #     continue

        features = get_token_features(token)
        features.update(sequence_features)
        features["sequence_id"] = row["seq_id"]
        features["token"] = token
        features["pos"] = pos
        features["cefr"] = cefr

        psy_retriever = (token, pos)
        if psy_retriever not in psy_feature_dict:
            psy_retriever = (lem, pos)

        psy_features = psy_feature_dict.get(
            psy_retriever, {"freq_pm": np.nan, "aoa": np.nan}
        )

        cefr_j_retriever = (token, pos)
        if cefr_j_retriever not in cefr_j_dict:
            cefr_j_retriever = (lem, pos)

        cefr_j = cefr_j_dict.get(cefr_j_retriever, np.nan)

        features["cefr_j"] = cefr_j
        features.update(psy_features)
        features.update(
            {
                "is_noun": pos == "NOUN",
                "is_verb": pos == "VERB",
                "is_adjective": pos == "ADJ",
                "is_adverb": pos == "ADV",
            }
        )
        features["label"] = lab

        yield features


def wrapped_process_row(in_q, out_q, psy_feature_dict, cefr_j_dict):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = lambda seq: pretokenized_to_doc(nlp, seq)

    received = in_q.get()
    while not isinstance(received, str):
        assert isinstance(received, tuple)
        row_i, row = received
        for feature_i, features in enumerate(
            process_row(row, nlp, psy_feature_dict, cefr_j_dict)
        ):
            features["row_i"] = row_i
            features["feature_i"] = feature_i
            out_q.put(features)

        received = in_q.get()
    else:
        assert received == "STOP"

    out_q.put("DONE")


def multiprocess_df(
    df: pd.DataFrame, psy_tsv_fpath: Path, cefr_j_tsv_fpath: Path, n_jobs: int = 12
) -> Generator[dict[str, Any], None, None]:
    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()

    psy_feature_dict = load_psy_feature_dict(psy_tsv_fpath)
    cefr_j_dict = load_cefr_j_dict(cefr_j_tsv_fpath)

    with mp.Pool(
        n_jobs,
        initializer=wrapped_process_row,
        initargs=(in_q, out_q, psy_feature_dict, cefr_j_dict),
    ) as pool:
        for row_i, row in df.iterrows():
            in_q.put((row_i, row))

        for _ in range(n_jobs):
            in_q.put("STOP")

        done_count = 0
        while done_count != n_jobs:
            received = out_q.get()
            if received == "DONE":
                done_count += 1
            else:
                yield received


def pretokenized_to_doc(nlp, seq: str) -> Doc:
    assert len(seq) != 0, f"Empty sequence: {seq}"
    return Doc(nlp.vocab, words=seq.split("|||"))


def convert_to_feature_df(
    original_df: pd.DataFrame, psy_tsv_fpath: Path, cefr_j_tsv_fpath: Path
) -> pd.DataFrame:
    feature_df = pd.DataFrame(
        (
            features
            for features in multiprocess_df(
                original_df, psy_tsv_fpath, cefr_j_tsv_fpath
            )
        )
    )
    feature_df.sort_values(by=["row_i", "feature_i"], inplace=True)

    return feature_df


@click.command()
@click.argument("tsv_fpath_str", type=click.Path(exists=True))
@click.argument("psy_fpath_str", type=click.Path(exists=True))
@click.argument("cefr_j_fpath_str", type=click.Path(exists=True))
def run_featurisation(
    tsv_fpath_str: str, psy_fpath_str: str, cefr_j_fpath_str: str
) -> None:
    tsv_fpath = Path(tsv_fpath_str)
    psy_tsv_fpath = Path(psy_fpath_str)
    cefr_j_tsv_fpath = Path(cefr_j_fpath_str)

    original_df = pd.read_csv(tsv_fpath, sep="\t", keep_default_na=False)
    feature_df = convert_to_feature_df(original_df, psy_tsv_fpath, cefr_j_tsv_fpath)

    output_fpath = tsv_fpath.parent / f"{tsv_fpath.stem}_features.tsv"
    feature_df.to_csv(output_fpath, sep="\t", index=False)


if __name__ == "__main__":
    run_featurisation()
