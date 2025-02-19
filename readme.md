# **Official Git Repository for COLING 2025 Paper**

**Paper Title**  
[**Does RAG Introduce Unfairness in LLMs? Evaluating Fairness in Retrieval-Augmented Generation Systems**](https://aclanthology.org/2025.coling-main.669/)

This repository contains the **source code** and **data** for our paper accepted at COLING 2025. It investigates whether retrieval methods and knowledge sources introduce biases or disparities in Large Language Models (LLMs) when using Retrieval-Augmented Generation (RAG) systems.

---

## **Table of Contents**
1. [Overview](#overview)  
2. [Installation and Environment](#installation-and-environment)  
3. [FlashRAG Modifications](#flashrag-modifications)  
4. [Data Preparation](#data-preparation)  
5. [Index Building](#index-building)  
6. [Evaluation](#evaluation)  
7. [Contact](#contact)

---

## **1. Overview** <a name="overview"></a>

In this project, we explore how the retrieval component in a RAG pipeline can lead to unfair or biased outcomes when combined with LLMs. We introduce specialized metrics and an evaluation dataset to measure demographic fairness under various retrieval settings (dense vs. sparse, different embedding models, etc.).

**Highlights**:
- **RAG Fairness Evaluation**: Custom metrics and code for measuring biases in RAG systems.
- **Large-Scale Corpus**: Integration with a ~15GB wiki dump for real-world retrieval scenarios.
- **Flexible Implementation**: Built atop [FlashRAG](https://github.com/codecodebear/FlashRAG-SCU) for modular retrieval and generation.

---

## **2. Installation and Environment** <a name="installation-and-environment"></a>

We provide a Conda environment file for convenience. **Ensure you have Conda** (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)).

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate my_rag_env
```
Note: Adjust the Conda environment name if it's specified differently in environment.yml.


## **3. FlashRAG Modification**<a name="flashrag-modifications"></a>

We introduced custom evaluation metrics and configuration options in FlashRAG to evaluate fairness in RAG systems. These changes may conflict with an existing FlashRAG installation.

Patched Files::
    * `flashrag/config/config.py`: add evaluation mode.
    * `flashrag/evaluator/metrics.py`: add runtime evalution code.
    * `flashrag/prompt/base_prompt.py`: debug for 'instruct' model.

Search for comments like:

```
#-----------------# For RAG Fairness Only #-----------------#

#-----------------------------------------------------------#
```

and apply or overwrite these files in your local FlashRAG setup. Reinstall FlashRAG or adjust your PYTHONPATH to use our modified version.



## **4. Data Preparation**<a name="data-preparation"></a>

1. Download Wiki Corpus

    ```
    wget https://huggingface.co/datasets/codecodebear/wiki_dump/resolve/main/wiki_dump.jsonl
    ```
    This 15GB file may take significant time to download.

2. Preprocessing

    * Run `preprocess.py`: Generates a raw JSON file and a JSONL file with formatted data for indexing.
    * Example usage:
        ```
        python preprocess.py
        ```
    * Adjust the script or arguments as needed for your dataset.

3. Fairness Evaluation Data

    * Run prepare_rag_fairness_data.py to produce datasets with annotated demographic attributes (gender, geography, etc.).
    * Outputs stored in data/trek_2022_fairness.

## **5. Index Building** <a name="index-building"></a>

We use [FlashRAG’s index-building guide](https://github.com/codecodebear/FlashRAG-SCU/blob/main/docs/building-index.md). Ensure your corpus is in JSONL format:

```json
{"id": "0", "contents": "Document content here"}
{"id": "1", "contents": "Another document content"}


For dense retrieval methods, especially the popular embedding models, we use faiss to build index. Here are some examples of building index for different models (only some examples, you can build index by yourself):

* Based on `e5-base-v2` embedding model:

```
CUDA_VISIBLE_DEVICES=7 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /path_to_your_model_file/e5-base-v2 \
    --corpus_path /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_formatted.jsonl \
    --save_dir /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat 
```

* Based on `e5-large-v2` embedding model:

```
CUDA_VISIBLE_DEVICES=7 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /path_to_your_model_file/e5-large-v2 \
    --corpus_path /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_formatted.jsonl \
    --save_dir /path_to_your_data_file/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index_e5_large \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat 
```

* Based on `bm25` retrieval method:

```
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path /path_to_your_data_file/trek_fair_2022_train_id_title_doc.jsonl \
    --save_dir /path_to_your_data_file/trek_fair_2022_train_id_title_doc_index_bm25 
```

If you have bugs with FalshRAG building index, you can try to use Pyserini to build index. Here is an example of building index using Pyserini:

```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /ssd/public_datasets/wiki/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index_bm25/temp \
  --index /ssd/public_datasets/wiki/trek_fair_2022_train_id_title_only_first_100w_clean_corpus_index_bm25/bm25 \
  --generator DefaultLuceneDocumentGenerator\
  --threads 1 
```

## **6. Evaluating** <a name="evaluation"></a>

To evaluate the model, you can use the following command:

For example:

```
python run_exp.py --method_name 'zero-shot' --split 'flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800' --dataset_name 'trek_2022_fairness' --gpu_id '7' --save_note 'zero-shot_flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800'
```

* The script references my_config.yaml for model paths, hyperparameters, etc.
* Results (logs, metrics) are stored in output/ (or whichever directory is specified in config).

## **7. Contact** <a name="contact"></a>

If you have any questions, issues, or suggestions, please reach out via:

Contact Email: xwu5@scu.edu

Thank you for your interest in our work! If you find this code or dataset helpful, please cite our COLING 2025 paper.

```
@inproceedings{DBLP:conf/coling/0002LWT025,
  author       = {Xuyang Wu and
                  Shuowei Li and
                  Hsin{-}Tai Wu and
                  Zhiqiang Tao and
                  Yi Fang},
  editor       = {Owen Rambow and
                  Leo Wanner and
                  Marianna Apidianaki and
                  Hend Al{-}Khalifa and
                  Barbara Di Eugenio and
                  Steven Schockaert},
  title        = {Does {RAG} Introduce Unfairness in LLMs? Evaluating Fairness in Retrieval-Augmented
                  Generation Systems},
  booktitle    = {Proceedings of the 31st International Conference on Computational
                  Linguistics, {COLING} 2025, Abu Dhabi, UAE, January 19-24, 2025},
  pages        = {10021--10036},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.coling-main.669/},
  timestamp    = {Tue, 28 Jan 2025 16:22:22 +0100},
  biburl       = {https://dblp.org/rec/conf/coling/0002LWT025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
