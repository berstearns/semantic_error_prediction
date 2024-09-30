"""Convenience utility for combining TSV-files"""

import click
import pandas as pd


@click.command()
@click.argument("in_fpaths", nargs=-1, type=click.Path(exists=True))
@click.argument("out_fpath", nargs=1, type=click.Path(exists=False))
@click.option("--header_missing", is_flag=True, show_default=True, default=False)
def combine_tsv(in_fpaths, out_fpath, header_missing):
    """Main function for combining TSV files."""

    if header_missing:
        header = None
    else:
        header = "infer"

    concat_df = pd.concat(
        (pd.read_csv(fp, sep="\t", header=header) for fp in in_fpaths)
    )

    concat_df.to_csv(out_fpath, sep="\t", index=False)


if __name__ == "__main__":
    combine_tsv()
