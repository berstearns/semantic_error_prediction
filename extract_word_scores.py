import json

from collections import Counter

import click
import pandas as pd


def create_counts(featurised_df: pd.DataFrame) -> dict[str, float]:
    tokens = featurised_df.token
    pos_tags = featurised_df.pos
    labels = featurised_df.label

    all_tagged_tokens = tuple(
        (t, p) for t, p, l in zip(tokens, pos_tags, labels) if l in ("true", "false")
    )

    tagged_tokens_counter = Counter(all_tagged_tokens)

    clicked_tagged_tokens = tuple(
        (t, p) for t, p, l in zip(tokens, pos_tags, labels) if l == "true"
    )
    clicked_tagged_tokens_counter = Counter(clicked_tagged_tokens)

    scores = {}
    for key, value in tagged_tokens_counter.items():
        if key not in clicked_tagged_tokens_counter:
            continue

        scores["|||".join(key)] = clicked_tagged_tokens_counter[key] / value

    return scores


@click.command()
@click.argument("tsv_fpath_list", nargs=-1)
def run_featurisation(tsv_fpath_list: list[str]) -> None:

    all_dfs = [pd.read_csv(fpath, sep="\t") for fpath in tsv_fpath_list]
    full_df = pd.concat(all_dfs)

    scores = create_counts(full_df)

    with open("count_scores.json", "w", encoding="utf-8") as j_file:
        json.dump(scores, j_file, indent=2)


if __name__ == "__main__":
    run_featurisation()



# python extract_word_scores.py train/train_features.tsv train/new_dev_features.tsv dev/all_dev_features.tsv
