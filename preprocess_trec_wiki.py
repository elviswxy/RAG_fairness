import argparse
from tqdm import tqdm
import re
import html
import spacy
import os
import json
import subprocess
from pathlib import Path
import shutil
import pandas as pd
import ir_datasets
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / 'data'


def load_corpus(dir_path):
    def iter_files(path):
        """Walk through all files located under a root path."""
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    yield os.path.join(dirpath, f)
        else:
            raise RuntimeError("Path %s is invalid" % path)

    def read_jsonl_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                corpus.append(json_data)

    all_files = [file for file in iter_files(dir_path)]
    corpus = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for file_path in all_files:
            executor.submit(read_jsonl_file, file_path)

    return corpus


def create_segments(doc_text, max_length, stride):
    doc_text = doc_text.strip()
    doc = nlp(doc_text)
    sentences = [sent.text.strip() for sent in doc.sents]
    segments = []

    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i : i + max_length])
        segments.append(segment)
        if i + max_length >= len(sentences):
            break

    return segments


def basic_process(title, text, id):
    title = html.unescape(title)
    text = html.unescape(text)
    text = text.strip()

    if "(disambiguation)" in title.lower():
        return None, None, None
    if "(disambiguation page)" in title.lower():
        return None, None, None
    # Take out List/Index/Outline pages (mostly links)
    if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", title):
        return None, None, None
    if text.startswith("REDIRECT") or text.startswith("redirect"):
        return None, None, None
    if text.endswith(". References."):
        text = text[: -len(" References.")].strip()

    text = re.sub("\{\{cite .*?\}\}", " ", text, flags=re.DOTALL)
    text = text.replace(r"TABLETOREPLACE", " ")
    text = text.replace(r"'''", " ")
    text = text.replace(r"[[", " ")
    text = text.replace(r"]]", " ")
    text = text.replace(r"{{", " ")
    text = text.replace(r"}}", " ")
    text = text.replace("<br>", " ")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("& amp;", "&")
    text = text.replace("nbsp;", " ")
    text = text.replace("formatnum:", "")

    # text = re.sub('<poem.*?</poem>', ' ', text, flags=re.DOTALL) # might have useful information?
    text = re.sub("<math.*?</math>", "", text, flags=re.DOTALL)
    text = re.sub("<chem.*?</chem>", "", text, flags=re.DOTALL)
    text = re.sub("<score.*?</score>", "", text, flags=re.DOTALL)

    # clean residual mess from xml dump that shouldn't have made its way here
    text = re.sub("\| ?item[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub("\| ?col[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub("\| ?row[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub("\| ?style= ?.*? ", " ", text)
    text = re.sub("\| ?bodystyle= ?.*? ", " ", text)
    text = re.sub("\| ?frame_?style= ?.*? ", " ", text)
    text = re.sub("\| ?data_?style= ?.*? ", " ", text)
    text = re.sub("\| ?label_?style= ?.*? ", " ", text)
    text = re.sub("\| ?headerstyle= ?.*? ", " ", text)
    text = re.sub("\| ?list_?style= ?.*? ", " ", text)
    text = re.sub("\| ?title_?style= ?.*? ", " ", text)
    text = re.sub("\| ?ul_?style= ?.*? ", " ", text)
    text = re.sub("\| ?li_?style= ?.*? ", " ", text)
    text = re.sub("\| ?border-style= ?.*? ", " ", text)
    text = re.sub('\|? ?style=".*?"', "", text)
    text = re.sub('\|? ?rowspan=".*?"', "", text)
    text = re.sub('\|? ?colspan=".*?"', "", text)
    text = re.sub('\|? ?scope=".*?"', "", text)
    text = re.sub('\|? ?align=".*?"', "", text)
    text = re.sub('\|? ?valign=".*?"', "", text)
    text = re.sub('\|? ?lang=".*?"', "", text)
    text = re.sub('\|? ?bgcolor=".*?"', "", text)
    text = re.sub("\|? ?bg=\#[a-z]+", "", text)
    text = re.sub('\|? ?width=".*?"', "", text)
    text = re.sub("\|? ?height=[0-9]+", "", text)
    text = re.sub("\|? ?width=[0-9]+", "", text)
    text = re.sub("\|? ?rowspan=[0-9]+", "", text)
    text = re.sub("\|? ?colspan=[0-9]+", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub("<.*?/>", "", text)
    text = re.sub("\|? ?align=[a-z]+", "", text)
    text = re.sub("\|? ?valign=[a-z]+", "", text)
    text = re.sub("\|? ?scope=[a-z]+", "", text)
    text = re.sub("&lt;ref&gt;.*?&lt;/ref&gt;", " ", text)
    text = re.sub("&lt;.*?&gt;", " ", text)
    text = re.sub("File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?", "", text)
    text = re.sub("Source: \[.*?\]", "", text)
    text = text.replace("Country flag|", "country:")
    text = text.replace("flag|", "country:")
    text = text.replace("flagicon|", "country:")
    text = text.replace("flagcountry|", "country:")
    text = text.replace("Flagu|", "country:")
    text = text.replace("display=inline", "")
    text = text.replace("display=it", "")
    text = text.replace("abbr=on", "")
    text = text.replace("disp=table", "")

    title = title.replace("\n", " ").replace("\t", " ")

    text = ' '.join(text.split(' ')[:160])

    return title, text, id


def split_list(lst, n):
    """Split a list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def single_worker(docs):
    results = []
    for item in tqdm(docs):
        title, text, id = basic_process(item[0], item[1], item[2])
        if title is None:
            continue
        # title = f'"{title}"'
        results.append((title, text, id))
    return results

def load_trec():
    documents = []
    dataset = ir_datasets.load("trec-fair/2022/train")
    docs_store = dataset.docs_store()
    for item in tqdm(dataset.docs_iter()):
        id = item.doc_id
        title = item.title
        text = item.text
        documents.append((title, text, id))
    return documents

def get_formatted_jsonl(jsonl_file_path):
    # jsonl_file_path = 'data/trek_fair_2022_train_id_title_only_first_100w_clean_corpus.jsonl'
    new_lines = []
    # Open the JSONL file and read it line by line
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the line as a JSON object
            json_obj = json.loads(line.strip())
            new_lines.append({"id": json_obj['id'], "contents": "Title: " + json_obj['title'] + " Text: " + json_obj['text']})

    with open(jsonl_file_path.replace(".jsonl", "_formatted.jsonl"), 'w') as f:
        for pair in new_lines:
            f.write(json.dumps(pair) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean wiki corpus file for indexing.")
    # parser.add_argument("--dump_path", type=str)
    parser.add_argument("--chunk_by", default="100w", choices=["100w", "sentence"], type=str)
    parser.add_argument("--seg_size", default=None, type=int)
    parser.add_argument("--stride", default=None, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_path", type=str, default=DATA_PATH/"trek_fair_2022_train_id_title_only_first_100w_clean_corpus_gpu7.jsonl")
    args = parser.parse_args()

    # extract wiki dump
    # temp_dir = os.path.join(Path(args.save_path).parent, "temp")
    # os.makedirs(temp_dir)

    # temp_dir = DATA_PATH
    

    # corpus = load_corpus(temp_dir)
    nlp = spacy.load("en_core_web_lg", disable=['tok2vec', 'tagger', 'ner','parser','attribute_ruler', 'lemmatizer'])

    print("Start pre-processing...")
    documents = load_trec()

    with Pool(processes=args.num_workers) as p:
        result_list = list(tqdm(p.imap(single_worker, split_list(documents, args.num_workers))))
    result_list = sum(result_list, [])

    with open(DATA_PATH/'trek_fair_2022_train_id_title_only_first_100w_clean_gpu7.json', "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

    
    # result_list = []
    # for item in documents:
    #     print(basic_process(item[0], item[1], item[2]))
    #     title, text, id = basic_process(item[0], item[1], item[2])
    #     if title is None:
    #         continue
    #     title = f'"{title}"'
    #     result_list.append((title, text, id))
    all_title = [item[0] for item in result_list]
    all_text = [item[1] for item in result_list]
    all_id = [item[2] for item in result_list]

    print("Start chunking...")
    idx = 0
    clean_corpus = []
    with open(args.save_path, "w", encoding="utf-8") as f:
        if args.chunk_by == "sentence":
            for doc in tqdm(nlp.pipe(all_text, n_process=args.num_workers, batch_size=10000), total=len(all_text)):
                title = all_title[idx]
                id = all_id[idx]
                idx += 1
                sentences = [sent.text.strip() for sent in doc.sents]
                segments = []
                for i in range(0, len(sentences), args.stride):
                    segment = " ".join(sentences[i : i + args.seg_size])
                    segments.append(segment)
                    if i + args.seg_size >= len(sentences):
                        break
                for segment in segments:
                    text = segment.replace("\n", " ").replace("\t", " ")
                    clean_corpus.append({"id": id, "title": title, "text": text})

        elif args.chunk_by == "100w":
            for doc in tqdm(nlp.pipe(all_text, n_process=4, batch_size=10000), total=len(all_text)):
                title = all_title[idx]
                id = all_id[idx]
                idx += 1
                segments = []
                word_count = 0
                segment_tokens = []
                for token in doc:
                    segment_tokens.append(token.text_with_ws)
                    if not token.is_space and not token.is_punct:
                        word_count += 1
                        if word_count == 100:
                            word_count = 0
                            segments.append("".join([token for token in segment_tokens]))
                            segment_tokens = []
                if word_count != 0:
                    for token in doc:
                        segment_tokens.append(token.text_with_ws)
                        if not token.is_space and not token.is_punct:
                            word_count += 1
                            if word_count == 100:
                                word_count = 0
                                segments.append("".join([token for token in segment_tokens]))
                                break
                if word_count != 0:
                    segments.append("".join([token for token in segment_tokens]))

                # for segment in segments:
                for segment in segments[:1]:
                    text = segment.replace("\n", " ").replace("\t", " ")
                    # clean_corpus.append({"id": id, "title": title, "text": text})
                    item = {"id": id, "title": title, "text": text}
                    f.write(json.dumps(item) + "\n")

    get_formatted_jsonl(args.save_path)

    # shutil.rmtree(temp_dir)

    # print("Start saving corpus...")
    # with open(args.save_path, "w", encoding="utf-8") as f:
    #     for idx, item in enumerate(clean_corpus):
    #         title = f"\"{item['title']}\""
    #         item = {"id": item["id"], "title": item['title'], "text": item["text"]}
    #         f.write(json.dumps(item) + "\n")
    print("Finish!")
