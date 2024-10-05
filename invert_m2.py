# (C) original file by Chris Bryant, modified by David Strohmaier

import argparse


# Invert M2 annotations (orig -> cor, cor -> orig)
def main(args):
    # Open M2 and output files
    with open(args.m2_file) as m2, open(args.out, "w") as out_m2:
        # Get the m2 block
        block = []
        # Loop through lines
        for line in m2:
            line = line.strip()
            # If there is a line, add it to the block and continue
            if line:
                block.append(line)
                continue
            # We have a complete block: Get orig sent and edits
            orig = block[0].split()[1:]
            edits = simplify_edits(block[1:])
            # Apply edits to get cor and add cor span to edits
            cor, edits = apply_edits(orig, edits)
            # Write cor to output
            out_m2.write(" ".join(["S"] + cor) + "\n")
            # Write edits
            for e in edits:
                # Format edit
                edit = [
                    " ".join(["A", str(e[3]), str(e[4])]),
                    e[6],
                    " ".join(e[5]),
                    "REQUIRED",
                    "-NONE-",
                    e[-1],
                ]
                out_m2.write("|||".join(edit) + "\n")
            out_m2.write("\n")
            # Reset the block
            block = []


# Input: A list of edit lines from an m2 file
# Output: A list of lists. Each sublist is an edit: [start, end, cat, cor, id]
def simplify_edits(edits):
    out_edits = []
    for e in edits:
        e = e.split("|||")
        span = e[0][2:].split()  # [2:] ignore the leading "A "
        start = int(span[0])
        end = int(span[1])
        cat = e[1]
        cor = e[2].split()
        id = e[-1]
        # Save the useful info as a list
        proc_edit = [start, end, cat, cor, id]
        # Save the proc_edit
        out_edits.append(proc_edit)
    return out_edits


# Input 1: The original tokenised sentence as a list of tokens
# Input 2: The edits in that sentence
def apply_edits(orig, edits):
    # Output corrected sentence (updated in place)
    cor = orig[:]
    offset = 0
    # Loop through edits
    for i, e in enumerate(edits):
        # Short circuit noops
        if e[2] == "noop":
            edits[i] = [-1, -1, ["-NONE-"], -1, -1, ["-NONE-"], "noop", e[4]]
            continue
        # Calculate the cor span of the edit
        c_start = e[0] + offset
        c_end = c_start + len(e[3])
        c_txt = cor[e[0] + offset : e[1] + offset]
        # Apply the edit
        cor[e[0] + offset : e[1] + offset] = e[3]
        # Update offset
        offset = offset - (e[1] - e[0]) + len(e[3])
        # Update the edit
        edits[i] = [e[0], e[1], e[3], c_start, c_end, c_txt, e[2], e[4]]
    return cor, edits


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("m2_file", help="Input M2 file")
    parser.add_argument(
        "-out", help="A path to where we save the combined m2 file.", required=True
    )
    args = parser.parse_args()
    main(args)
