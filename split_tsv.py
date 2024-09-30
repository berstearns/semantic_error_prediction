"""Convenience utility for splitting TSV-files into train and test data."""

from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_fnc(df: pd.DataFrame, selected: str, test_prop: float, seed: int):
    unique_vals = df[selected].unique()
    train_vals, _ = train_test_split(
        unique_vals, test_size=test_prop, random_state=seed
    )

    train_df = df[df[selected].isin(train_vals)]
    test_df = df[~df[selected].isin(train_vals)]
    return train_df, test_df


@click.command()
@click.argument("input_fpath", type=click.Path(exists=True))
@click.argument("output_dir_str", type=click.Path(exists=True))
@click.option("--create_dev", default=True, type=bool)
@click.option("--split_by_selected", default="", type=str)
@click.option("--test_prop", default=0.1, type=float)
@click.option("--seed", default=1848, type=int)
def split_tsv(
    input_fpath: str,
    output_dir_str: str,
    create_dev: bool,
    split_by_selected: str,
    test_prop: float,
    seed: int,
) -> None:
    """Split tsv-file into multiple files for train/testing.

    :param input_fpath: str-path to input tsv-file
    :param output_dir_str: str-path to output directory
    :param create_dev: whether to create a dev split in addition to a train-test split
    :param split_by_selected: whether to split based on value in selected column
    :param test_prop: proportion of test set
    :param seed: random seed used for splitting
    :returns:

    """
    output_dir = Path(output_dir_str)

    original_df = pd.read_csv(input_fpath, sep="\t", keep_default_na=False)

    if split_by_selected:
        split_fnc = lambda df: split_by_fnc(df, split_by_selected, test_prop, seed)
    else:
        split_fnc = lambda df: train_test_split(
            df, test_size=test_prop, random_state=seed
        )

    traindev_df, test_df = split_fnc(original_df)

    if create_dev:
        train_df, dev_df = split_fnc(traindev_df)

        dev_path = Path(output_dir) / "dev.tsv"
        dev_df.to_csv(dev_path, sep="\t", index=False)
    else:
        train_df = traindev_df

    train_path = Path(output_dir) / "train.tsv"
    train_df.to_csv(train_path, sep="\t", index=False)

    test_path = Path(output_dir) / "test.tsv"
    test_df.to_csv(test_path, sep="\t", index=False)


if __name__ == "__main__":
    split_tsv()  # pylint: disable=no-value-for-parameter
