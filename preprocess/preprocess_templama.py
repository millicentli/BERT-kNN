"""
Preprocesses the TempLAMA dataset to be in the same format as the other LAMA datasets

Copy the Squad dataset -- minimum requirements:
- obj_label (the masked token)
- id (just copy over the id)
- sub_label (name of the dataset)

Format:
"masked_sentences": [""], "obj_label": "", "id":, "", "sub_label", ""

python preprocess_templama.py --file_name /private/home/millicentli/BERT-kNN/data/TempLAMA/test.json
"""

import argparse
import json
import os
import re

from transformers import BertTokenizer

def get_sample(tokenizer, string):
    tokenized_text = tokenizer.tokenize(string)
    indexed_string = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, help="the file to preprocess into the correct format")
    parser.add_argument("--with_dates", action='store_true')
    parser.add_argument("--filter_multi_word", action='store_true')
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    save_dir = "/".join(args.file_name.split("/")[:-1])
    # breakpoint()
    if not args.with_dates and not args.filter_multi_word:
        f = open(os.path.join(save_dir, "TempLAMA.json"), "w")
        output_file_counts = open(os.path.join(save_dir, "TempLAMA_counts.txt"), "w")
        output_file_counts = None
    elif not args.with_dates and args.filter_multi_word:
        f = open(os.path.join(save_dir, "TempLAMA_filtered.json"), "w")
        output_file_counts = open(os.path.join(save_dir, "TempLAMA_filtered_counts.txt"), "w")
    elif args.with_dates and not args.filter_multi_word:
        f = open(os.path.join(save_dir, "TempLAMA_with_dates.json"), "w")
        output_file_counts = open(os.path.join(save_dir, "TempLAMA_with_dates_counts.txt"), "w")
    else:
        f = open(os.path.join(save_dir, "TempLAMA_with_dates_filtered.json"), "w")
        output_file_counts = None
        # output_file_counts = open(os.path.join(save_dir, "TempLAMA_with_dates_filtered_counts.txt"), "w")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    total = 0
    used = 0
    for line in open(args.file_name, 'r'):
        total += 1
        used += 1

        loaded = json.loads(line)

        # masked_sentences
        query = loaded['query']
        query = query.replace("_X_", "[MASK]")

        # obj_label
        label = loaded['answer'][0]['name']

        # sub_label
        sub_indices = [x.span() for x in re.finditer('([A-ZÀ-ÖØ][a-zA-ZÀ-ÖØ-öø-ÿ]+)', query)]
        if len(sub_indices) > 2:
            sub_label = query[sub_indices[0][0] : sub_indices[len(sub_indices) - 2][1]]
        else:
            sub_label = query[sub_indices[0][0] : sub_indices[0][1]]

        # skip if the word is too big
        if args.filter_multi_word:
            res = get_sample(tokenizer, label)
            if len(res) > 1:
                used -= 1
                continue

        # date
        date = loaded['date']
        
        # id
        id = loaded['id']

        # if it's TempLAMA, we're just going to use the context (like for Squad)
        if args.with_dates:
            context = f"{date}. {query}"
            new_line = {"masked_sentences": [context], "obj_label": label, "id": id, "date": date, "sub_label": sub_label, "type": "TempLAMA"}
        else:
            new_line = {"masked_sentences": [query], "obj_label": label, "id": id, "date": date, "sub_label": sub_label, "type": "TempLAMA"}

        
        json.dump(new_line, f)
        f.write('\n')
    if args.filter_multi_word:
        output_file_counts.write(f"Here are the total counts: {total}\n")
        output_file_counts.write(f"Here are the num. of queries kept: {used}\n")
        output_file_counts.close()

    f.close()


if __name__ == "__main__":
    main()