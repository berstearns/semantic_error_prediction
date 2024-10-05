# (C) original file by Chris Bryant, modified by David Strohmaier

import argparse
import errant
import json
import re
import spacy
from bisect import bisect
from operator import itemgetter
from string import punctuation
from functools import partial
from multiprocessing import Pool


def get_offsets(orig_sent):
    return type(orig_sent[0])


# Convert BEA2019 Shared Task style JSON to M2.
def main():
    # Parse command line args
    args = parse_args()
    print("Loading resources...")
    # Punctuation normalisation dictionary
    norm_dict = {
        "’": "'",
        "´": "'",
        "‘": "'",
        "′": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "˝": '"',
        "¨": '"',
        "„": '"',
        "『": '"',
        "』": '"',
        "–": "-",
        "—": "-",
        "―": "-",
        "¬": "-",
        "、": ",",
        "，": ",",
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "ِ": " ",
        "\u200b": " ",
    }
    norm_dict = {ord(k): v for k, v in norm_dict.items()}
    # Open output M2 file
    # out_m2 = open(args.out, "w")

    print("Preprocessing files...")
    # Open the file
    with open(args.json_file) as data:
        # Process each line

        func = partial(process_line, args=args, norm_dict=norm_dict)
        with Pool(3) as pool:
            all_outputs = pool.map(func, data)

    all_m2 = []
    all_cefr = []
    for output in all_outputs:
        all_m2.extend(output["m2"])
        all_cefr.extend(output["cefr"])

    combined_m2 = "".join(all_m2)
    with open(args.out + ".m2", "w") as out_file:
        out_file.write(combined_m2)

    with open(args.out + ".cefr.json", "w") as j_file:
        json.dump(all_cefr, j_file)


def process_line(line, args, norm_dict):
    # Load Tokenizer and other resources
    nlp = spacy.load("en_core_web_sm")
    # Load Errant
    annotator = errant.load("en", nlp)

    cefr_levels = []
    to_write = []
    line = json.loads(line)
    # Normalise certain punctuation in the text
    text = line["text"].translate(norm_dict)
    # Store the sentences and edits for all annotators here
    coder_dict = {}
    # Loop through the annotator ids and their edits
    for coder, edits in line["edits"]:
        # Add the coder to the coder_dict if needed
        if coder not in coder_dict:
            coder_dict[coder] = []
        # Split the essay into paras and update and normalise the char edits
        para_info = get_paras(text, edits, norm_dict)
        # Loop through the paras and edits
        for orig_para, para_edits in para_info:
            # Remove unnecessary whitespace from para and update char edits

            cefr_levels.append(line["cefr"][:2])
            orig_para, para_edits = clean_para(orig_para, para_edits)
            if not orig_para:
                continue  # Ignore empty paras
            # Convert character edits to token edits
            orig_para = nlp(orig_para)
            para_edits = get_token_edits(orig_para, para_edits, nlp)
            # Split the paragraph into sentences, if needed, and update tok edits
            sents = get_sents(orig_para, para_edits, args.sents)
            # Save the sents in the coder_dict
            coder_dict[coder].extend(sents)
    # Get the sorted coder ids
    coder_ids = sorted(coder_dict.keys())
    # Loop through the sentences for the first coder
    for sent_id, sent in enumerate(coder_dict[0]):
        # Write the original sentence to the output M2 file
        to_write.append("S " + " ".join(sent["orig"]) + "\n")
        # Annotate the original sentence with spacy
        orig = annotator.parse(" ".join(sent["orig"]))
        # Loop through the coders
        for id in coder_ids:
            # Annotate the corrected sentence with spacy and get the gold edits
            cor = annotator.parse(" ".join(coder_dict[id][sent_id]["cor"]))
            gold_edits = coder_dict[id][sent_id]["edits"]
            # Gold edits
            if args.gold:
                # Make sure edits are ordered by orig start and end offsets.
                gold_edits = sorted(gold_edits, key=itemgetter(0))  # Start
                gold_edits = sorted(gold_edits, key=itemgetter(1))  # End
                proc_edits = []
                # Loop through the gold edits.
                for gold_edit in gold_edits:
                    # Format the edit for errant import
                    gold_edit = gold_edit[:2] + gold_edit[-2:] + [gold_edit[2]]
                    # Detection edits (never minimised)
                    if gold_edit[-1] == "D":
                        gold_edit = annotator.import_edit(
                            orig, cor, gold_edit, min=False, old_cat=args.old_cats
                        )
                    # Correction edits
                    else:
                        gold_edit = annotator.import_edit(
                            orig, cor, gold_edit, not args.no_min, args.old_cats
                        )
                        # Ignore edits that have been minimised to nothing
                        if gold_edit.o_start == gold_edit.o_end and not gold_edit.c_str:
                            continue
                    # Save the edit in proc edits
                    proc_edits.append(gold_edit)
                # If there are no edits, write an explicit noop edit.
                if not proc_edits:
                    to_write.append(noop_edit(id) + "\n")
                # Write the edits to the output M2 file
                for edit in proc_edits:
                    to_write.append(edit.to_m2(id) + "\n")
            # Auto edits
            elif args.auto:
                auto_edits = annotator.annotate(orig, cor, args.lev, args.merge)
                # If there are no edits, write an explicit noop edit.
                if not auto_edits:
                    to_write.append(noop_edit(id) + "\n")
                # Write the edits to the output M2 file
                for edit in auto_edits:
                    to_write.append(edit.to_m2(id) + "\n")
        # Write new line after each sentence when we reach last coder.
        to_write.append("\n")

    return {"m2": "".join(to_write), "cefr": cefr_levels}


# Parse command line args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BEA2019 Shared Task style JSON to M2 format.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [-h] (-auto | -gold) [options] json_file -out <out_name>",
    )
    parser.add_argument(
        "json_file", help="Path to a JSON file, one JSON essay per line."
    )
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument(
        "-auto", help="Extract edits automatically.", action="store_true"
    )
    type_group.add_argument(
        "-gold", help="Use existing edit alignments.", action="store_true"
    )
    parser.add_argument("-out", help="The output filepath.", required=True)
    parser.add_argument(
        "-sents", help="The text is already sentence tokenised.", action="store_true"
    )
    parser.add_argument(
        "-no_min", help="Do not minimise edit spans (gold only).", action="store_true"
    )
    parser.add_argument(
        "-old_cats",
        help="Preserve old error types (gold only); i.e. turn off the classifier.",
        action="store_true",
    )
    parser.add_argument(
        "-lev", help="Align using standard Levenshtein.", action="store_true"
    )
    parser.add_argument(
        "-merge",
        help="Choose a merging strategy for automatic alignment.\n"
        "rules: Use a rule-based merging strategy (default)\n"
        "all-split: Merge nothing: MSSDI -> M, S, S, D, I\n"
        "all-merge: Merge adjacent non-matches: MSSDI -> M, SSDI\n"
        "all-equal: Merge adjacent same-type non-matches: MSSDI -> M, SS, D, I",
        choices=["rules", "all-split", "all-merge", "all-equal"],
        default="rules",
    )
    args = parser.parse_args()
    return args


# Input 1: An essay string.
# Input 2: A list of character edits in the essay
# Input 3: A string normalisation dictionary for unusual punctuation etc.
# Output: A list of paragraph strings and their edits [(para, edits), ...]
def get_paras(text, edits, norm_dict):
    para_info = []
    # Loop through all sequences between newlines
    # (?:.{0,25}\n+){,5}.+\n*(?:.{0,25}(?:\n|$))*
    reg_expression = "(?:\s*.{0,25}\s*\n+){,5}.+\n*(?:\s*.{0,25}\s*(?:\n|$))*"
    for para in re.finditer(reg_expression, text):  # Merging in short paragraphs
        # Other option ".+\n*.+\n*" -- using two paragraphs at a time

        para_edits = []
        # Keep track of correction spans (not detection spans)
        cor_spans = []
        # Loop through the edits: [start, end, cor, <type>]; <type> may be optional
        for edit in edits:
            # Find edits that fall inside this paragraph
            if edit[0] >= para.start(0) and edit[1] <= para.end(0):
                # Adjust offsets and add C or D type for correction or detection
                new_edit = [
                    edit[0] - para.start(0),
                    edit[1] - para.start(0),
                    "C",
                    edit[2],
                ]
                if edit[2] == None:
                    new_edit[2] = "D"
                # Normalise the string if its a correction edit
                if new_edit[2] == "C":
                    new_edit[3] = edit[2].translate(norm_dict)
                    # Preserve the error type if it is already known
                    if len(edit) == 4:
                        new_edit[2] = edit[3]
                    # Save the span in cor_spans
                    cor_spans.append(new_edit[:2])
                # Save the edit
                para_edits.append(new_edit)
            # Activate this switch to see cross para edits that are ignored, if any.
        #            elif edit[0] >= para.start(0) and edit[0] <= para.end(0) and \
        #                    edit[1] > para.end(0):
        #                print(text); print(edit)
        # Remove overlapping detection edits from the list (for FCE only)
        new_para_edits = []
        # Loop through the new normalised edits again
        for edit in para_edits:
            # Find detection edits
            if edit[2] == "D":
                # Boolean if the edit overlaps with a correction
                overlap = False
                # Loop through cor_spans
                for start, end in cor_spans:
                    # Check whether there are any cor edits inside this det edit.
                    if (start != end and start >= edit[0] and end <= edit[1]) or (
                        start == end and start > edit[0] and end < edit[1]
                    ):
                        overlap = True
                # If there is an overlap, ignore the detection edit
                if overlap:
                    continue
            new_para_edits.append(edit)
        # Save the para and the para edits
        para_info.append((para.group(0), new_para_edits))
    return para_info


# Input 1: An untokenized paragraph string.
# Input 2: A list of character edits in the input string.
# Output 1: The same as Input 1, except unnecessary whitespace has been removed.
# Output 2: The same as Input 2, except character edit spans have been updated.
def clean_para(para, edits):
    # Replace all types of whitespace with a space
    para = re.sub("\s", " ", para)
    # Find any sequence of 2 adjacent whitespace characters
    # NOTE: Matching 2 at a time lets us preserve edits between multiple whitespace.
    match = re.search("  ", para)
    # While there is a match...
    while match:
        # Find the index where the whitespace starts.
        ws_start = match.start()
        # Remove 1 of the whitespace chars.
        para = para[:ws_start] + para[ws_start + 1 :]
        # Update affected edits that start after ws_start
        for edit in edits:
            # edit = [start, end, ...]
            if edit[0] > ws_start:
                edit[0] -= 1
            if edit[1] > ws_start:
                edit[1] -= 1
        # Try matching again
        match = re.search("  ", para)
    # Remove leading whitespace, if any.
    if para.startswith(" "):
        para = para.lstrip()
        # Subtract 1 from all edits.
        for edit in edits:
            # edit = [start, end, ...]
            # "max" used to prevent negative index
            edit[0] = max(edit[0] - 1, 0)
            edit[1] = max(edit[1] - 1, 0)
    # Remove leading/trailing whitespace from character edit spans
    for edit in edits:
        # Ignore insertions
        if edit[0] == edit[1]:
            continue
        # Get the orig text
        orig = para[edit[0] : edit[1]]
        # Remove leading whitespace and update span
        if orig.startswith(" "):
            edit[0] += 1
        if orig.endswith(" "):
            edit[1] -= 1
    # Return para and new edit spans.
    return para, edits


# Input 1: A spacy paragraph
# Input 2: A list of character edits in the input string.
# Input 3: A spacy processing object
# Output: A list of token edits that map to exact tokens.
def get_token_edits(para, edits, nlp):
    # Get the character start and end offsets of all tokens in the para.
    tok_starts, tok_ends = get_all_tok_starts_and_ends(para)
    prev_tok_end = 0
    overlap_edit_ids = []
    # edit = [start, end, cat, cor]
    for edit in edits:
        # Set cor to orig string if this is a detection edit
        if edit[3] == None:
            edit[3] = para.text[edit[0] : edit[1]]
        # Convert the character spans to token spans.
        span = convert_char_to_tok(edit[0], edit[1], tok_starts, tok_ends)
        # If chars do not map cleanly to tokens, extra processing is needed.
        if len(span) == 4:
            # Sometimes token expansion creates overlapping edits. Keep track of this.
            if span[0] < prev_tok_end:
                overlap_edit_ids.append(edits.index(edit))
                continue
            # When span len is 4, span[2] and [3] are the new char spans.
            # Use these to expand the edit to match token boundaries.
            left = para.text[span[2] : edit[0]]
            right = para.text[edit[1] : span[3]]
            # Add this new info to cor.
            edit[3] = (left + edit[3] + right).strip()
        # Keep track of prev_tok_end
        prev_tok_end = span[1]
        # Change char span to tok span
        edit[0] = span[0]
        edit[1] = span[1]
        # Tokenise correction edits
        if edit[2] != "D":
            edit[3] = " ".join([tok.text for tok in nlp(edit[3].strip())])
        # Set detection edits equal to the tokenised original
        elif edit[2] == "D":
            edit[3] = " ".join([tok.text for tok in para[edit[0] : edit[1]]])
    # Finally remove any overlap token edits from the edit list (rare)
    for id in sorted(overlap_edit_ids, reverse=True):
        del edits[id]
    return edits


# Input: A spacy paragraph
# Output: A list of character start and end positions for each token in the input.
def get_all_tok_starts_and_ends(spacy_doc):
    tok_starts = []
    tok_ends = []
    for tok in spacy_doc:
        tok_starts.append(tok.idx)
        tok_ends.append(tok.idx + len(tok.text))
    return tok_starts, tok_ends


# Input 1: A char start position
# Input 2: A char end position
# Input 3: All the char token start positions in the paragraph
# Input 4: All the char token end positions in the paragraph
# Output: The char start and end position now in terms of tokens.
def convert_char_to_tok(start, end, all_starts, all_ends):
    # If the start and end span is the same, the edit is an insertion.
    if start == end:
        # Special case: Pre-First token edits.
        if not start or start <= all_starts[0]:
            return [0, 0]
        # Special case: Post-Last token edits.
        elif start >= all_ends[-1]:
            return [len(all_starts), len(all_starts)]
        # General case 1: Edit starts at the beginning of a token.
        elif start in all_starts:
            return [all_starts.index(start), all_starts.index(start)]
        # General case 2: Edit starts at the end of a token.
        elif start in all_ends:
            return [all_ends.index(start) + 1, all_ends.index(start) + 1]
        # Problem case: Edit starts inside 1 token.
        else:
            # Expand character span to nearest token boundary.
            if start not in all_starts:
                start = all_starts[bisect(all_starts, start) - 1]
            if end not in all_ends:
                end = all_ends[bisect(all_ends, end)]
            # Keep the new character spans as well
            return [all_starts.index(start), all_ends.index(end) + 1, start, end]
    # Character spans match complete token spans.
    elif start in all_starts and end in all_ends:
        return [all_starts.index(start), all_ends.index(end) + 1]
    # Character spans do NOT match complete token spans.
    else:
        # Expand character span to nearest token boundary.
        if start not in all_starts:
            start = all_starts[bisect(all_starts, start) - 1]
        if end not in all_ends:
            nearest = bisect(all_ends, end)
            # Sometimes the end is a char after the last token.
            # In this case, just use the last tok boundary.
            if nearest >= len(all_ends):
                end = all_ends[-1]
            else:
                end = all_ends[bisect(all_ends, end)]
        # Keep the new character spans as well
        return [all_starts.index(start), all_ends.index(end) + 1, start, end]


# Input 1: A SpaCy original paragraph Doc object.
# Input 2: A list of edits in that paragraph.
# Input 3: A flag whether the text is already sentence tokenised or not
# Output: A list of dictionaries. Each dict has 3 keys: orig, cor, edits
# Sentences are split according to orig only. Edits map orig to cor.
def get_sents(orig, edits, sent_tokenised):
    # ignoring sent boundaries

    sent_list = []
    orig, cor, edits = prepare_sent_edits_output(orig, edits)
    out_dict = {"orig": orig, "cor": cor, "edits": edits}
    sent_list.append(out_dict)
    return sent_list


# Input 1: A tokenized original sentence.
# Input 2: The edits in that sentence.
# Output 1: The tokenized corrected sentence from these edits.
# Output 2: The edits, now containing the tok span of cor_str in cor_sent.
def prepare_sent_edits_output(orig, edits):
    orig = [tok.text for tok in orig]
    cor = orig[:]
    offset = 0
    for edit in edits:
        # edit = [orig_start, orig_end, cat, cor]
        cor_toks = edit[3].split()
        cor[edit[0] + offset : edit[1] + offset] = cor_toks
        cor_start = edit[0] + offset
        cor_end = cor_start + len(cor_toks)
        offset = offset - (edit[1] - edit[0]) + len(cor_toks)
        # Save cor offset
        edit.extend([cor_start, cor_end])
    return orig, cor, edits


# Input: A coder id
# Output: A noop edit; i.e. text contains no edits
def noop_edit(id=0):
    return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(id)


# Run the program
if __name__ == "__main__":
    main()
