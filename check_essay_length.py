import re
import json
import string

import click


def max_essay_length(essay: str):
    reg_expression = "(?:\s*.{0,25}\s*\n+){,5}.+\n*(?:\s*.{0,25}\s*(?:\n|$))*"

    return max(
        len(essay[match.start():match.end()])
        for match in re.finditer(reg_expression, essay)
    )

    # return max(len(para) for para in essay.split("\n"))


def check_json_file(fp_str: str, limit: int = 2000):
    with open(fp_str, "r") as json_file:
        for i, line in enumerate(json_file):
            if line in string.whitespace:
                continue

            essay = json.loads(line)["text"]
            max_len = max_essay_length(essay)
            if max_len > limit:
                print(f"Long essay at {fp_str} ({i}): {max_len}")


@click.command()
@click.argument("json_files", nargs=-1)
def check(json_files: list[str]):
    for fp_str in json_files:
        check_json_file(fp_str)


if __name__ == "__main__":
    check()
