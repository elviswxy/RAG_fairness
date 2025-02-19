import json
import pandas as pd
import re
import os
import json
import yaml
import shutil
import csv
from tqdm import tqdm
from flashrag.dataset.dataset import Dataset, Item
from flashrag.evaluator import Evaluator
from flashrag.config import Config
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / 'data' 
OUTPUT_PATH = CURRENT_DIR / 'output'

class EvalOnlyDataset(Dataset):
    def __init__(self, path):
        self.data = self._load_data(path)
    
    def _load_data(self, dataset_path):
        """Load data from the provided dataset_path or directly download the file(TODO)."""

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            for item_dict in results:
                item = Item(item_dict)
                data.append(item)
        return data

def get_existed_results(csv_file_path):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        return df['path'].tolist()
    else:
        return []

# main is used to evaluate all the models in the output directory
# and save the results to a CSV file
# two modes are supported: "update" and "new"
# "update" mode will append the new results to the existing CSV file
# "new" mode will overwrite the existing CSV file
# dataset_name is the name of the dataset: "trek_2022_fairness" or "trek_2022_geo"
# csv_file_path is the path to the CSV file
def main(dataset_name, csv_file_path, mode="update"):
    if mode == "update":
        exist_paths = get_existed_results(csv_file_path)
    else:
        exist_paths = []
    results = []
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for dir in tqdm(dirs):
            log_dir = os.path.join(OUTPUT_PATH, dir)
            if dataset_name in log_dir:
                for root, dirs, files in os.walk(log_dir):
                    if mode == "update" and root in exist_paths:
                        continue
                    if 'intermediate_data.json' in files:
                        json_path = os.path.join(root, 'intermediate_data.json')
                        yaml_path = os.path.join(root, 'config.yaml')
                        config = Config(yaml_path, {"evaluation_mode": True})
                        config["save_metric_score"] = False
                        config["save_intermediate_data"] = False
                        # config["evaluation_mode"] = True
                        if 'priorityfairness' not in config["metrics"]:
                            config["metrics"].append('priorityfairness')
                        if 'optionem' not in config["metrics"]:
                            config["metrics"].append('optionem')
                        # if 'bleu' not in config["metrics"]:
                        #     config["metrics"].append('bleu')
                        if 'rouge-1' not in config["metrics"]:
                            config["metrics"].append('rouge-1')
                        if 'rouge-2' not in config["metrics"]:
                            config["metrics"].append('rouge-2')
                        if 'rouge-l' not in config["metrics"]:
                            config["metrics"].append('rouge-l')
                        if 'mrr' not in config["metrics"]:
                            config["metrics"].append('mrr')
                        if 'judgeeval' not in config["metrics"]:
                            config["metrics"].append('judgeeval') 
                        
                        data = EvalOnlyDataset(json_path)
                        evaluator = Evaluator(config)
                        results.append(['_'.join(config['save_note'].split('_')[1:]), config['save_note'].split('_')[0], config['retrieval_method'], config['retrieval_topk']] + list(evaluator.evaluate(data).values()) + [root])
                        # print(evaluator.evaluate(data))
                    else:
                        print(f"intermediate_data.json not found in: {root}\n")
                        # shutil.rmtree(root)
            
    # Write the results to a CSV file
    if mode == "update":
        with open(csv_file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(results)
    else:
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Dataset', 'Method', 'retriever', 'retriever_num', 'em', 'f1', 'acc', 'precision', 'recall', 'position_male_ratio', 'position_female_ratio', 'optionem', 'rouge-1', 'rouge-2', 'rouge-l', 'mrr_male@1','mrr_female@1','mrr_male@all','mrr_female@all', 'judger_true_m_fair', 'judger_true_f_fair', 'judger_false_m_fair', 'judger_false_f_fair', 'judge_true_em', 'judge_false_em', 'path'])
            writer.writerows(results)

    print(f"Results saved to {csv_file_path}")

# get_specific_results is used to get the results of a specific model
def get_specific_results(path):
    root = path
    json_path = os.path.join(root, 'intermediate_data.json')
    yaml_path = os.path.join(root, 'config.yaml')
    config = Config(yaml_path)
    config["save_metric_score"] = False
    config["save_intermediate_data"] = False
    config["evaluation_mode"] = True
    if 'priorityfairness' not in config["metrics"]:
        config["metrics"].append('priorityfairness')
    if 'mrr' not in config["metrics"]:
        config["metrics"].append('mrr')
    # if 'optionem' not in config["metrics"]:
        # config["metrics"].append('optionem')
    if 'mrr' not in config["metrics"]:
        config["metrics"].append('judgeeval')
    data = EvalOnlyDataset(json_path)
    evaluator = Evaluator(config)
    print(['_'.join(config['save_note'].split('_')[1:]), config['save_note'].split('_')[0], config['retrieval_method'], config['retrieval_topk'], root] + list(evaluator.evaluate(data).values()))

# results saved to output directory
if __name__ == '__main__':
    main('trek_2022_fairness', DATA_PATH/'offline_gender_evaluation_results_new.csv', "update")
    # main('trek_2022_fairness', DATA_PATH/'offline_gender_evaluation_results_new.csv', "new")
    # main('trek_2022_geo', OUTPUT_PATH/'offline_geo_evaluation_results.csv')
    # get_specific_results(OUTPUT_PATH/'trek_2022_fairness_2024_09_07_03_58_naive_flashrag_trek_2022_fairness_all_relevant_male_female_4800_e5r5')


