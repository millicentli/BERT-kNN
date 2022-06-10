import os
import sqlite3
import unicodedata
import transformers
import json
from nltk.corpus import stopwords
import string
import argparse
import time
from tqdm import tqdm

from spacy.lang.en import English
import multiprocessing
from multiprocessing.pool import ThreadPool

nlp = English()

class DocDB(object):
    """Sqlite backed document storage.
    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def masking(args):
    db = args["db"]
    db_ids = args["db_ids"]
    save_idx = args["save_idx"]
    ids_per_dump = args["ids_per_dump"]
    stop_words = args["stop_words"]
    save_files_sentences = args["save_files_sentences"]
    save_files_labels = args["save_files_labels"]
    save_files_dbids = args["save_files_dbids"]
    model_type = args["model_type"]
    curr_idx = save_idx * ids_per_dump

    if model_type == "BERT":
        transformer_tok = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_type == "ROBERTA":
        transformer_tok = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    else:
        transformer_tok = transformers.T5Tokenizer.from_pretrained("t5-base")
    
    tokenizer = nlp.tokenizer

    print(f"Save_ids: {save_idx}. Curr_idx: {curr_idx}.")
    start = time.time()
    num_sents = len(db_ids[curr_idx:])
    if num_sents < curr_idx + ids_per_dump:
        end = curr_idx + num_sents
    else:
        end = curr_idx + ids_per_dump

    for idx in range(curr_idx, end):
        _id = db_ids[idx]
        raw_text = db.get_doc_text(_id)
        raw_list = raw_text.split("\n")
        raw_list = list(filter(None, raw_list))
        for line in raw_list:
            sentences = list(filter(None, line.split(".")))
            sents = ""
            labels = ""
            dbids = ""
            for sentence in sentences:
                sentence = sentence.strip()
                tokens = tokenizer(sentence)
                for idx, token in enumerate(tokens):
                    # print("Here's sents:", sents)
                    token = token.string.strip().lower()
                    if model_type == "BERT":
                        if token in transformer_tok.vocab and token not in stop_words and token not in string.punctuation:
                            masked_sentences = \
                                (" ".join([tokens[i].string.strip().lower() for i in range(idx)]) + " [MASK] " + " ".join([tokens[i].string.strip().lower() for i in range(idx + 1, len(tokens))]) + ".")
                            sents = "\n".join(filter(None, [sents, masked_sentences]))
                            labels = "\n".join(filter(None, [labels, token]))
                            dbids = "\n".join(filter(None, [dbids, _id]))
                    # TODO: deal with the problem -- the fact that a lot of these vocab words don't exist in T5 -- what do you do?
                    else:
                        if token in transformer_tok.get_vocab() and token not in stop_words and token not in string.punctuation:
                            masked_sentences = \
                                (" ".join([tokens[i].string.strip().lower() for i in range(idx)]) + " <extra_id_0> " + " ".join([tokens[i].string.strip().lower() for i in range(idx + 1, len(tokens))]) + ".")
                            sents = "\n".join(filter(None, [sents, masked_sentences]))
                            labels = "\n".join(filter(None, [labels, token]))
                            dbids = "\n".join(filter(None, [dbids, _id]))
            if sents != "":
                save_files_sentences.write(sents)
                save_files_sentences.write("\n")
                save_files_labels.write(labels)
                save_files_labels.write("\n")
                save_files_dbids.write(dbids)
                save_files_dbids.write("\n")

    save_files_sentences.close()
    save_files_labels.close()
    save_files_dbids.close()
    end = time.time()
    print(f"Time elapsed: {end - start}. Save_ids: {save_idx}. Curr_idx: {curr_idx}.")

def main(args):
    num_dumps = 100

    if not args.bert and not args.t5 and not args.roberta:
        raise ValueError("Either BERT must be selected, RoBERTa must be selected, or T5 must be selected!")

    # wikipedia data base
    path_db = args.path_db_wikipedia_drqa
    db = DocDB(path_db)
    db_ids = db.get_doc_ids()
    ids_per_dump = int(len(db_ids)/num_dumps) + 1

    if args.bert:
        save_dir = "/private/home/millicentli/BERT-kNN/DrQA/data/wikidump_batched_bert/"
        save_file = save_dir + "dump_"
        model_type = "BERT"
    elif args.roberta:
        save_dir = "/private/home/millicentli/BERT-kNN/DrQA/data/wikidump_batched_roberta/"
        save_file = save_dir + "dump_"
        model_type = "RoBERTa"
    else:
        save_dir = "/private/home/millicentli/BERT-kNN/DrQA/data/wikidump_batched_t5/"
        save_file = save_dir + "dump_"
        model_type = "T5"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_files_sentences = {}
    save_files_labels = {}
    save_files_dbids = {}
    for n in range(len(db_ids) // ids_per_dump + 1):
        if not os.path.exists(save_dir + str(n) + "_dbids.txt"):
            save_files_dbids[n] = open(save_file + str(n) + "_dbids.txt", "w")
            save_files_sentences[n] = open(save_file + str(n) + "_sentences.txt", "w")
            save_files_labels[n] = open(save_file + str(n) + "_labels.txt", "w")
    # if not os.path.exists(save_dir + str(99) + "_dbids.txt"):
    #     save_files_dbids[99] = open(save_file + str(99) + "_dbids.txt", "w")
    #     save_files_sentences[99] = open(save_file + str(99) + "_sentences.txt", "w")
    #     save_files_labels[99] = open(save_file + str(99) + "_labels.txt", "w")

    stop_words = set(stopwords.words('english'))

    # Initialize Threadpool
    num_threads = multiprocessing.cpu_count()
    # Cap it at half the number of dumps for efficiency
    if num_threads > 50:
        num_threads = 50
    # num_threads = 1
    pool = ThreadPool(num_threads)

    print(f"Starting multithreading with {num_threads} threads")
    print("Start masking!")

    args = [{
        "db": db,
        "db_ids": db_ids,
        "save_idx": idx,
        "ids_per_dump": ids_per_dump,
        "stop_words": stop_words,
        "save_files_sentences": save_files_sentences[idx],
        "save_files_labels": save_files_labels[idx],
        "save_files_dbids": save_files_dbids[idx],
        "model_type": model_type
    } for idx in range(len(db_ids) // ids_per_dump + 1)]
    
    # args = [{
    #     "db": db,
    #     "db_ids": db_ids,
    #     "save_idx": 99,
    #     "ids_per_dump": ids_per_dump,
    #     "stop_words": stop_words,
    #     "save_files_sentences": save_files_sentences[99],
    #     "save_files_labels": save_files_labels[99],
    #     "save_files_dbids": save_files_dbids[99],
    #     "model_type": model_type
    # }]

    pool.map(masking, args)

    pool.close()
    pool.join()
    print("All tasks have been finished!")

    id_dict = {}
    for d in range(len(db_ids) // ids_per_dump + 1):
        with open(save_file + str(d) + "_dbids.txt") as f:
            idx = 0
            num_ent = 0
            for line in f:
                line = line.strip().lower()
                if line not in id_dict:
                    id_dict[line] = [d, num_ent, idx, idx]
                    num_ent += 1
                else:
                    id_dict[line][3] += 1
                idx += 1

    with open(save_file + "dict_id_idcs.json", "w") as f:
        json.dump(id_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_db_wikipedia_drqa",
        default=0,
        type=str,
        required=True,
        help="Path_drqa"
    )
    parser.add_argument(
        "--bert",
        action="store_true"
    )
    parser.add_argument(
        "--t5",
        action="store_true"
    )
    parser.add_argument(
        "--roberta",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
    # main(args.path_db_wikipedia_drqa)
