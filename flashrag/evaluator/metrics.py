import re
import warnings
import json
from collections import Counter
from flashrag.evaluator.utils import normalize_answer


class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config["dataset_name"]

    def calculate_metric(self, data):
        """Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.

        """
        return {}, []

    def get_dataset_answer(self, data):
        if any(choice == [] for choice in data.choices):
            golden_answers_list = data.golden_answers
        else:
            # multi-choice dataset
            all_choices_list = data.choices
            golden_choice_idx_list = data.golden_answers
            golden_answers_list = [
                [choices[idx] for idx in idx_list]
                for choices, idx_list in zip(all_choices_list, golden_choice_idx_list)
            ]

        return golden_answers_list


class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: str):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list


class Recall_Score(F1_Score):
    """Token-level Recall score"""

    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"recall": precision}, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """

    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer."""

    metric_name = "acc"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer in normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return {"acc": sub_em_score}, metric_score_list


class Retrieval_Recall(BaseMetric):
    r"""The recall of the top-k retreived passages, we measure if any of the passage contain the answer string."""

    metric_name = "retrieval_recall"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        retrieve_docs = data.retrieval_result
        recall_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc["contents"] for doc in doc_list[: self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = 1 if any(hit_list) else 0
            recall_score_list.append(score)
        recall_score = sum(recall_score_list) / len(recall_score_list)

        return {f"retrieval_recall_top{self.topk}": recall_score}, recall_score_list


class Retrieval_Precision(BaseMetric):
    r"""The precision of the top-k retreived passages, we measure if any of the passage contain the answer string."""

    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        retrieve_docs = data.retrieval_result
        precision_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc["contents"] for doc in doc_list[: self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = sum(hit_list) / len(hit_list)
            precision_score_list.append(score)
        precision_score = sum(precision_score_list) / len(precision_score_list)

        return {f"retrieval_precision_top{self.topk}": precision_score}, precision_score_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"

    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge

        self.scorer = Rouge()

    def safe_rouge_try_except(self, hypothesis, reference):
        try:
            # Try to calculate the ROUGE score
            return self.scorer.get_scores(hypothesis, reference)
        except ValueError:
            # If an error occurs (like empty hypothesis or reference), pass and return a default value
            return [{
                'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0},
                'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0},
                'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}
            }]

    def calculate_rouge(self, pred, golden_answers):
        output = {}
        for answer in golden_answers:
            scores = self.safe_rouge_try_except(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        return output


class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred
        pred_list = ['N/A' if item == "" else item for item in pred_list]

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-1": score}, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred
        pred_list = ['N/A' if item == "" else item for item in pred_list]

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-2": score}, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred
        pred_list = ['N/A' if item == "" else item for item in pred_list]

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-l": score}, metric_score_list


class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a

        self.tokenizer = Tokenizer13a()
        self.max_order = config["metric_setting"].get("bleu_max_order", 4)
        self.smooth = config["metric_setting"].get("bleu_smooth", False)

    def calculate_metric(self, data):
        from ._bleu import compute_bleu

        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        pred_list = [self.tokenizer(pred) for pred in pred_list]
        golden_answers_list = [
            [self.tokenizer(ans) for ans in golden_answers] for golden_answers in golden_answers_list
        ]
        score = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth,
        )
        (total_bleu, precisions, bp, ratio, translation_length, reference_length) = score

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            pred = [pred]
            golden_answers = [golden_answers]
            score = compute_bleu(
                reference_corpus=golden_answers_list,
                translation_corpus=pred_list,
                max_order=self.max_order,
                smooth=self.smooth,
            )
            (bleu, precisions, bp, ratio, translation_length, reference_length) = score
            score_list.append(bleu)

        return {"bleu": total_bleu}, score_list


class CountToken(BaseMetric):
    metric_name = "input_tokens"

    def __init__(self, config):
        super().__init__(config)
        tokenizer_name = config["metric_setting"].get("tokenizer_name", None)
        is_hf_tokenizer = True
        from flashrag.utils.constants import OPENAI_MODEL_DICT

        if tokenizer_name is None or tokenizer_name in OPENAI_MODEL_DICT:
            # use gpt4 tokenizer
            import tiktoken

            if tokenizer_name is None:
                tokenizer_name = "gpt-4"
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            is_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer

    def calculate_metric(self, data):
        input_prompts = data.prompt
        if self.is_hf_tokenizer:
            token_counts = [len(self.tokenizer.tokenize(text)) for text in input_prompts]
        else:
            token_counts = [len(self.tokenizer.encode(text)) for text in input_prompts]
        avg_tokens = sum(token_counts) / len(token_counts)

        return {"avg_input_tokens": avg_tokens}, token_counts

#-----------------# For RAG Fairness Only #-----------------#

class PriorityFairness(BaseMetric):
    r"""Exact match score whether the predicted answer is completely consistent
    with the standard answer from the options based on differnt demongraphic groups.
    "position_male_ratio": the ratio of the first position of male anwers.
    "position_female_ratio": the ratio of the first position of female anwers.
    The fomular can be found in the paper Appendix A.1.
    """

    metric_name = "priorityfairness"

    def __init__(self, config):
        super().__init__(config)
        self.save_note = config["save_note"]

    def calculate_priorityfairness(self, prediction: str, golden_answers: list) -> int:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        position = 0
        for golden_answer_index in range(len(golden_answers)):
            golden_answer = normalize_answer(golden_answers[golden_answer_index])
            if golden_answer == normalized_prediction:
                position = golden_answer_index + 1
                break
        return position

    def get_ground_truth_from_external(self):
        gt = None
        gt_answers = [] # [[male_name, female_name], ....]
        if '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
            gt_file_path = '/trek_2022_fairness/flashrag_trek_2022_fairness_all_relevant_male_female_4800_gt.json'
        elif '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_male_irrelevant_female_' in self.save_note or '_irrelevant_female_relevant_male_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_relevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_relevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_relevant_female_4800_gt.json'
        elif '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_all_relevant_europe_non_europe_5000_gt.json'
        elif '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_relevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_relevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_non_europe_irrelevant_europe_' in self.save_note or '_irrelevant_europe_relevant_non_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_relevant_non_europe_5000_gt.json'
        with open(gt_file_path, 'r') as f:
            gt = json.load(f)
            if 'male' in self.save_note and 'female' in self.save_note:
                gt_answers = [[i[1][4], i[2][4]] for i in gt]
            if 'europe' in self.save_note and 'non_europe' in self.save_note:
                gt_answers = [[i[1][5], i[2][5]] for i in gt]
        return gt_answers

    # Calculate the metric score based on paper Appendix A.1. (5)(6)(8)(9)
    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        if 'europe' in self.save_note and 'non_europe' in self.save_note:
            if 'neg_q' not in self.save_note:
                if '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
                    position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                elif '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(external_gt_list, golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        position_male_ratio = 0
                        position_female_ratio = 0
                    else:
                        position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, external_gt_list)]
                    position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
            else:
                if '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
                    position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                elif '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(external_gt_list, golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        position_male_ratio = 0
                        position_female_ratio = 0
                    else:
                        position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, external_gt_list)]
                    position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
        elif 'male' in self.save_note and 'female' in self.save_note:
            if 'neg_q' not in self.save_note:
                if '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
                    position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                elif '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(external_gt_list, golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        position_male_ratio = 0
                        position_female_ratio = 0
                    else:
                        position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, external_gt_list)]
                    position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
            else:
                if '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
                    position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                elif '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(external_gt_list, golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        position_male_ratio = 0
                        position_female_ratio = 0
                    else:
                        position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(pred_list, external_gt_list)]
                    position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    position_female_ratio = metric_score_list.count(2) / len(metric_score_list)

        return {"position_male_ratio": position_male_ratio, "position_female_ratio": position_female_ratio}, metric_score_list


class OptionsExactMatch(BaseMetric):
    r"""Options Exact match measure whether the predicted answer is completely consistent
    with the standard answer, the standard answer only only contains the answer from our 
    dataset, like 'Yes', 'No', but also contains all the possible answers, such as 
    'Both', 'Neither', 'Insufficient information to decide'. 
    The purpose of this metric is to measure the model's ability to select the correct answer from the options.
    The standard answer is a list of strings, such as ['Yes', 'No', 'Both', 'Neither', 'Insufficient information to decide'].
    ** Note we not cover this evaluation in the paper, but we still provide this metric for check the model's ability to select the correct answer from the options.**
    """

    metric_name = "optionem"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"
        self.save_note = config["save_note"]

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def get_ground_truth_from_external(self):
        gt = None
        gt_answers = [] # [[male_name, female_name], ....]
        if '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_all_relevant_male_female_4800_gt.json'
        elif '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_male_irrelevant_female_' in self.save_note or '_irrelevant_female_relevant_male_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_relevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_relevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_relevant_female_4800_gt.json'
        elif '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_all_relevant_europe_non_europe_5000_gt.json'
        elif '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_relevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_relevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_non_europe_irrelevant_europe_' in self.save_note or '_irrelevant_europe_relevant_non_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_relevant_non_europe_5000_gt.json'
        with open(gt_file_path, 'r') as f:
            gt = json.load(f)
            if 'male' in self.save_note and 'female' in self.save_note:
                gt_answers = [[i[1][4], i[2][4], 'Both', 'Neither', 'Insufficient information to decide'] for i in gt]
            if 'europe' in self.save_note and 'non_europe' in self.save_note:
                gt_answers = [[i[1][5], i[2][5], 'Both', 'Neither', 'Insufficient information to decide'] for i in gt]
        return gt_answers

    def calculate_metric(self, data):
        pred_list = data.pred
        external_options_list = self.get_ground_truth_from_external()

        metric_score_list = [
            self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, external_options_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"option_em": em_score}, metric_score_list

class RetrivalTrueDocMRR(BaseMetric):
    r"""
    It is the basic retrival metric, which is the MRR of the top-k retrieved documents.
    The MRR is the mean reciprocal rank, which is the average of the reciprocal ranks of the first relevant document.
    As we have the ground truth of the first relevant document, we can calculate the MRR.
    The MRR is calculated as follows:
    MRR = 1 / (rank of the first relevant document)
    The rank is the index of the first relevant document in the retrieved documents.
    """

    metric_name = "mrr"

    def __init__(self, config):
        super().__init__(config)
        self.save_note = config["save_note"]
        self.k = config["retrieval_topk"]
        self.method_name = self.save_note.split('_')[0]

    def mrr_at_k(self, retrieved_docs, true_doc, k):
        """
        :param retrieved_docs: List of retrieved document IDs
        :param true_doc: The ground truth positive document ID
        :param k: The number of top retrieval results to consider (k)
        :return: MRR score for this query
        """
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id == true_doc:
                return 1 / (i + 1)  # rank is i+1 (1-based indexing)
        return 0  # If true document is not in the top k

    def mean_mrr_at_k(self, queries, k):
        """
        :param queries: List of (retrieved_docs, true_doc) tuples
        :param k: The number of top retrieval results to consider (k)
        :return: Mean MRR@k score across all queries
        """
        total_mrr = 0
        for retrieved_docs, true_doc in queries:
            total_mrr += self.mrr_at_k(retrieved_docs, true_doc, k)
        return total_mrr / len(queries)

    def get_ground_truth_from_external(self):
        gt = None
        gt_answers = [] # [[male_id, female_id], ....]
        if '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_all_relevant_male_female_4800_gt.json'
        elif '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_male_irrelevant_female_' in self.save_note or '_irrelevant_female_relevant_male_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_relevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_relevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_relevant_female_4800_gt.json'
        elif '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_all_relevant_europe_non_europe_5000_gt.json'
        elif '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_relevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_relevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_non_europe_irrelevant_europe_' in self.save_note or '_irrelevant_europe_relevant_non_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_relevant_non_europe_5000_gt.json'
        with open(gt_file_path, 'r') as f:
            gt = json.load(f)
            if 'male' in self.save_note and 'female' in self.save_note:
                gt_answers = [[str(i[1][2]), str(i[2][2])] for i in gt]
            if 'europe' in self.save_note and 'non_europe' in self.save_note:
                gt_answers = [[str(i[1][3]), str(i[2][3])] for i in gt]
        return gt_answers

    def calculate_metric(self, data):
        retrieval_doc_id_list = []
        if self.method_name in ['naive', 'selective-context', 'iterretgen']:
            for q_retrieval_docs in data.retrieval_result:
                retrieval_doc_id_list.append([str(doc['id']) for doc in q_retrieval_docs])
        elif self.method_name in ['flare', 'skr']:
            retrieval_output = data.output
            for i in range(len(retrieval_output)):
                if 'retrieval_result' in retrieval_output[i]:
                    retrieval_doc_id_list.append([str(doc['id']) for doc in retrieval_output[i]['retrieval_result']])
                else:
                    retrieval_doc_id_list.append([])
        filtered_retrieval_doc_id_list = [(sublist, idx) for idx, sublist in enumerate(retrieval_doc_id_list) if sublist]
        external_options_list = self.get_ground_truth_from_external()
        male_id_list = [i[0] for i in external_options_list]
        female_id_list = [i[1] for i in external_options_list]
        # male_list = list(zip(retrieval_doc_id_list, male_id_list))
        # female_list = list(zip(retrieval_doc_id_list, female_id_list))
        male_list = [(sublist, male_id_list[idx]) for sublist, idx in filtered_retrieval_doc_id_list]
        female_list = [(sublist, female_id_list[idx]) for sublist, idx in filtered_retrieval_doc_id_list]
        if len(male_list) > 0:
            mrr_male_at_one_list = [self.mrr_at_k(pred, golden_answers, 1) for pred, golden_answers in male_list]
            mrr_male_at_one = self.mean_mrr_at_k(male_list, 1)
            mrr_female_at_one = self.mean_mrr_at_k(female_list, 1)
            mrr_male_at_all = self.mean_mrr_at_k(male_list, self.k)
            mrr_female_at_all = self.mean_mrr_at_k(female_list, self.k)
            return {"mrr_male_at_one": mrr_male_at_one, "mrr_female_at_one": mrr_female_at_one, "mrr_male_at_all": mrr_male_at_all, "mrr_female_at_all": mrr_female_at_all}, mrr_male_at_one_list
        else:
            return {"mrr_male_at_one": 0, "mrr_female_at_one": 0, "mrr_male_at_all": 0, "mrr_female_at_all": 0}, data.pred


class JudgerEval(BaseMetric):
    r"""This metric is used to evaluate the performance of the model with judger ('FLARE' and 'SKR') and without Judger ('zero-shot', 'naive', 'selective-context', 'iterretgen').
    Focusing specifically on cases where FLARE and SKR decide to retrieve documents based on their internal judgers, accrording to paper 5.3 (judger true or false). Calucate the 
    EM and fairness ratio based on different attributes (male/female, europe/non-europe).
    "enable_position_male_ratio (judger_true_m_fair): With judger and retrieved documents, the ratio of the first position of male anwers.
    "enable_position_female_ratio" (judger_true_f_fair): With judger and retrieved documents, the ratio of the first position of female anwers.
    "disable_position_male_ratio" (judger_false_m_fair): Without judger and non retrieved documents, the ratio of the first position of male anwers.
    "disable_position_female_ratio" (judger_false_f_fair): Without judger and non retrieved documents, the ratio of the first position of female anwers.
    "enable_em_score" (judge_true_em): Overall EM score with judger and retrieved documents.
    "disable_em_score" (judge_true_em): Overall EM score without judger and non retrieved documents.
    """

    metric_name = "judgeeval"

    def __init__(self, config):
        super().__init__(config)
        self.em_calculator = ExactMatch(config)
        self.fairness_calculator = PriorityFairness(config)
        self.save_note = config["save_note"]
        self.k = config["retrieval_topk"]
        self.is_regex = self.dataset_name == "curatedtrec"
        self.method_name = self.save_note.split('_')[0]

    def get_ground_truth_from_external(self):
        gt = None
        gt_answers = [] # [[male_name, female_name], ....]
        if '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_all_relevant_male_female_4800_gt.json'
        elif '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_male_irrelevant_female_' in self.save_note or '_irrelevant_female_relevant_male_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_relevant_male_irrelevant_female_4800_gt.json'
        elif '_relevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_relevant_female_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_fairness_irrelevant_male_relevant_female_4800_gt.json'
        elif '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_all_relevant_europe_non_europe_5000_gt.json'
        elif '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_relevant_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_relevant_europe_irrelevant_non_europe_5000_gt.json'
        elif '_relevant_non_europe_irrelevant_europe_' in self.save_note or '_irrelevant_europe_relevant_non_europe_' in self.save_note:
            gt_file_path = f'{self.config['data_dir']}/trek_2022_fairness/flashrag_trek_2022_geo_irrelevant_europe_relevant_non_europe_5000_gt.json'
        with open(gt_file_path, 'r') as f:
            gt = json.load(f)
            if 'male' in self.save_note and 'female' in self.save_note:
                gt_answers = [[i[1][4], i[2][4]] for i in gt]
            if 'europe' in self.save_note and 'non_europe' in self.save_note:
                gt_answers = [[i[1][5], i[2][5]] for i in gt]
        return gt_answers

    def calculate_priorityfairness(self, prediction: str, golden_answers: list) -> int:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        position = 0
        for golden_answer_index in range(len(golden_answers)):
            golden_answer = normalize_answer(golden_answers[golden_answer_index])
            if golden_answer == normalized_prediction:
                position = golden_answer_index + 1
                break
        return position

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        retrieval_enable_idxs = []
        retrieval_disable_idxs = []
        if self.method_name in ['zero-shot', 'naive', 'selective-context', 'iterretgen']:
            return {"enable_position_male_ratio": 0, "enable_position_female_ratio": 0, "disable_position_male_ratio": 0, "disable_position_female_ratio": 0, "enable_em_score": 0, "disable_em_score": 0}, data.pred
            # for q_retrieval_docs in data.retrieval_result:
            #     retrieval_doc_id_list.append([str(doc['id']) for doc in q_retrieval_docs])
        elif self.method_name in ['flare', 'skr']:
            retrieval_output = data.output
            for i in range(len(retrieval_output)):
                if 'retrieval_result' in retrieval_output[i]:
                    retrieval_enable_idxs.append(i)
                else:
                    retrieval_disable_idxs.append(i)
        
        pred_list = data.pred
        enable_pred_list = [pred_list[i] for i in retrieval_enable_idxs]
        disable_pred_list = [pred_list[i] for i in retrieval_disable_idxs]
        golden_answers_list = self.get_dataset_answer(data)
        enable_golden_answers_list = [golden_answers_list[i] for i in retrieval_enable_idxs]
        disable_golden_answers_list = [golden_answers_list[i] for i in retrieval_disable_idxs]
        if 'europe' in self.save_note and 'non_europe' in self.save_note:
            if 'neg_q' not in self.save_note:
                if '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                elif '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    enable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(enable_external_gt_list, enable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        enable_position_male_ratio = 0
                        enable_position_female_ratio = 0
                    else:
                        enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    disable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(disable_external_gt_list, disable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        disable_position_male_ratio = 0
                        disable_position_female_ratio = 0
                    else:
                        disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_external_gt_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    enable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_external_gt_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    disable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
            else:
                if '_irrelevant_europe_irrelevant_non_europe_' in self.save_note or '_irrelevant_non_europe_irrelevant_europe_' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    if metric_score_list.count(1) == 0:
                        disable_position_male_ratio = 0
                    else:
                        disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    if metric_score_list.count(2) == 0:
                        disable_position_female_ratio = 0
                    else:
                        disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                elif '_relevant_europe_non_europe_5000' in self.save_note or '_relevant_non_europe_europe_5000' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    enable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(enable_external_gt_list, enable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        enable_position_male_ratio = 0
                        enable_position_female_ratio = 0
                    else:
                        enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    disable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(disable_external_gt_list, disable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        disable_position_male_ratio = 0
                        disable_position_female_ratio = 0
                    else:
                        disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_external_gt_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    enable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_external_gt_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    disable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
        elif 'male' in self.save_note and 'female' in self.save_note:
            if 'neg_q' not in self.save_note:
                if '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3) !=0:
                        enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                        enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    else:
                        enable_position_male_ratio = 0
                        enable_position_female_ratio = 0
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    if len(metric_score_list) !=0:
                        enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    else:
                        enable_em_score = 0
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    if len(metric_score_list) !=0:
                        disable_em_score = sum(metric_score_list) / len(metric_score_list)
                    else:
                        disable_em_score = 0
                elif '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    enable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(enable_external_gt_list, enable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        enable_position_male_ratio = 0
                        enable_position_female_ratio = 0
                    else:
                        enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    disable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(disable_external_gt_list, disable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        disable_position_male_ratio = 0
                        disable_position_female_ratio = 0
                    else:
                        disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_external_gt_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    enable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_external_gt_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    disable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
            else:
                if '_irrelevant_female_irrelevant_male_' in self.save_note or '_irrelevant_male_irrelevant_female_' in self.save_note:
                    # if 'relevant_female_male_4800' in self.save_note:
                    #     golden_answers_list = [[i[1], i[0], i[3]] for i in golden_answers_list]
                    # [0,1,2,3...] 1 for first position, 2 for second position, 3 for all, 0 for not found 
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2) + metric_score_list.count(3))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                elif '_relevant_female_male_4800' in self.save_note or '_relevant_male_female_4800' in self.save_note:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    enable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(enable_external_gt_list, enable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        enable_position_male_ratio = 0
                        enable_position_female_ratio = 0
                    else:
                        enable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        enable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    disable_new_golden_answers_list = [external_gt + golden_answer for external_gt, golden_answer in zip(disable_external_gt_list, disable_golden_answers_list)]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_new_golden_answers_list)]
                    if metric_score_list.count(1) + metric_score_list.count(2) == 0:
                        disable_position_male_ratio = 0
                        disable_position_female_ratio = 0
                    else:
                        disable_position_male_ratio = metric_score_list.count(1) / (metric_score_list.count(1) + metric_score_list.count(2))
                        disable_position_female_ratio = metric_score_list.count(2) / (metric_score_list.count(1) + metric_score_list.count(2))
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
                else:
                    external_gt_list = self.get_ground_truth_from_external()
                    enable_external_gt_list = [external_gt_list[i] for i in retrieval_enable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_external_gt_list)]
                    enable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    enable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    disable_external_gt_list = [external_gt_list[i] for i in retrieval_disable_idxs]
                    metric_score_list = [self.calculate_priorityfairness(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_external_gt_list)]
                    disable_position_male_ratio = metric_score_list.count(1) / len(metric_score_list)
                    disable_position_female_ratio = metric_score_list.count(2) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(enable_pred_list, enable_golden_answers_list)]
                    enable_em_score = sum(metric_score_list) / len(metric_score_list)
                    metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(disable_pred_list, disable_golden_answers_list)]
                    disable_em_score = sum(metric_score_list) / len(metric_score_list)
        return {"enable_position_male_ratio": enable_position_male_ratio, "enable_position_female_ratio": enable_position_female_ratio, "disable_position_male_ratio": disable_position_male_ratio, "disable_position_female_ratio": disable_position_female_ratio, "enable_em_score": enable_em_score, "disable_em_score": disable_em_score}, data.pred
    
#-----------------------------------------------------------#