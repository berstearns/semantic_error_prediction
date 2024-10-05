import json

from csv import DictWriter
from pathlib import Path
from functools import partial

import click
import spacy
from spacy.tokens import Doc


SELECTED_LABELS = {"R:VERB", "R:NOUN", "R:ADJ", "R:ADV"}
SELECTED_SPACY_TAGS = {"VERB", "NOUN", "ADJ", "ADV"}


def create_label_seq(corrections: list[str], seq_length: int) -> list[str]:
    label_seq = ["false" for _ in range(seq_length)]

    for corr in corrections:
        corr_parts = corr.split("|||")

        if len(corr_parts) < 2:
            print(corr_parts)

        corr_label = corr_parts[1]
        if corr_label not in SELECTED_LABELS:
            continue
        leading, start, end = corr_parts[0].split()
        assert leading == "A"
        for i in range(int(start), int(end)):
            # label_seq[i] = corr_label
            label_seq[i] = "true"

    return label_seq


def parse_block(m2_block: list[str]):
    token_seq = m2_block[0].split()
    assert token_seq[0] == "S"  # check for special leading token
    seq_length = len(token_seq) - 1

    label_seq = create_label_seq(m2_block[1:], seq_length)

    if len(token_seq) == 1:
        print(f"empty m2 block: {m2_block}")
    return token_seq[1:], label_seq



def subselect_label_seq(token_seq: list[str], label_seq: list[str], nlp) -> list[str]:
    pos_tags = [t.pos_ for t in nlp("|||".join(token_seq))]

    new_label_seq = []

    assert len(pos_tags) == len(label_seq)
    for tag, label in zip(pos_tags, label_seq):
        # we discard some true labels

        if tag in SELECTED_SPACY_TAGS:
            new_label_seq.append(label)
        else:
            new_label_seq.append("NONE")
    return new_label_seq


@click.command()
@click.argument("m2_fpath_str", type=click.Path(exists=True))
@click.argument("json_fpath_str", type=click.Path(exists=True))
@click.argument("out_fpath", type=click.Path())
def convert_m2_tsv(m2_fpath_str: str, json_fpath_str: str, out_fpath) -> None:
    m2_fpath = Path(m2_fpath_str)
    json_path = Path(json_fpath_str)

    with open(json_path, "r", encoding="utf-8") as j_file:
        cefr_levels = json.load(j_file)

    def tokenized_list_to_doc(seq: str, nlp) -> Doc:
        assert len(seq) != 0, f"Empty sequence: {seq}"
        return Doc(nlp.vocab, words=seq.split("|||"))

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = partial(tokenized_list_to_doc, nlp=nlp)

    field_names = ["seq_id", "token_seq", "label_seq", "cefr"]
    with open(m2_fpath) as in_file, open(out_fpath, "w", newline="") as out_file:
        writer = DictWriter(out_file, fieldnames=field_names, delimiter="\t")
        writer.writeheader()

        m2_block = []
        block_count = 0
        for line in in_file:
            # print('So far so good')
            line = line.strip()
            if line != "":
                m2_block.append(line)
                continue

            # print(m2_block)
            token_seq, label_seq = parse_block(m2_block)

            if not token_seq:
                m2_block = []
                block_count += 1
                # in rare cases the whole sentence gets deleted
                continue

            label_seq = subselect_label_seq(token_seq, label_seq, nlp)

            row_dict = {
                "seq_id": f"{m2_fpath.name}_{block_count}",
                "token_seq": "|||".join(token_seq),
                "label_seq": "|||".join(label_seq),
                "cefr": cefr_levels[block_count],
            }

            writer.writerow(row_dict)
            m2_block = []
            block_count += 1


if __name__ == "__main__":
    convert_m2_tsv()

#  LocalWords:  sm
