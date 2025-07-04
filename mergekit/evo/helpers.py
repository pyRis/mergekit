# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union
import sys
import json

import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

from mergekit.evo.config import TaskConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

# Add the qu-du-tasks directory to the path to import the evaluation modules
sys.path.append('/Users/pyris/research/IRIT/qu-du-tasks')
from inference_dataset import InferenceDataset
from inference_tasks.query_expansion import QueryExpansion
from inference_tasks.query_reformulation import QueryReformulation
from inference_tasks.fact_verification import FactVerification
from inference_tasks.summarization import Summarization


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _eval_model(
    model_path: str,
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Custom evaluation function using qu-du-tasks evaluation system
    """
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    
    # Map task names to actual task objects and their label lists
    task_mapping = {
        # Query expansion tasks
        "query_expansion_gov2": (QueryExpansion(name="gov2", shot="zero_shot", setting="in-domain"), []),
        "query_expansion_trec_robust": (QueryExpansion(name="trec_robust", shot="zero_shot", setting="in-domain"), []),
        "query_expansion_trec_covid": (QueryExpansion(name="trec_covid", shot="zero_shot", setting="in-domain"), []),
        "query_expansion_fire": (QueryExpansion(name="fire", shot="zero_shot", setting="in-domain"), []),
        "query_expansion_query2doc": (QueryExpansion(name="query2doc", shot="zero_shot", setting="in-domain"), []),
        "query_expansion_trec_cast": (QueryExpansion(name="trec_cast", shot="zero_shot", setting="in-domain"), []),
        "query_expansion_trec_web": (QueryExpansion(name="trec_web", shot="zero_shot", setting="in-domain"), []),
        
        # Query reformulation tasks
        "query_reformulation_codec": (QueryReformulation(name="codec", shot="zero_shot", setting="in-domain"), []),
        "query_reformulation_qrecc": (QueryReformulation(name="qrecc", shot="zero_shot", setting="in-domain"), []),
        "query_reformulation_canard": (QueryReformulation(name="canard", shot="zero_shot", setting="in-domain"), []),
        "query_reformulation_trec_cast": (QueryReformulation(name="trec_cast", shot="zero_shot", setting="in-domain"), []),
        "query_reformulation_gecor": (QueryReformulation(name="gecor", shot="zero_shot", setting="in-domain"), []),
        
        # Fact verification tasks
        "fact_verification_fever": (FactVerification(name="fever", shot="zero_shot", setting="in-domain"), ["support", "refute"]),
        "fact_verification_climate_fever": (FactVerification(name="climate_fever", shot="zero_shot", setting="in-domain"), ["support", "refute", "disputed", "not enough information"]),
        "fact_verification_scifact": (FactVerification(name="scifact", shot="zero_shot", setting="in-domain"), ["support", "refute"]),
        
        # Summarization tasks
        "summarization_cnndm": (Summarization(name="cnndm", shot="zero_shot", setting="in-domain"), []),
        "summarization_xsum": (Summarization(name="xsum", shot="zero_shot", setting="in-domain"), []),
        "summarization_wikisum": (Summarization(name="wikisum", shot="zero_shot", setting="in-domain"), []),
        "summarization_multinews": (Summarization(name="multinews", shot="zero_shot", setting="in-domain"), []),
    }
    
    results = {}
    max_input_len = 1792
    max_output_len = 256
    test_batch_size = 2
    
    for task_config in tasks:
        task_name = task_config.name
        if task_name not in task_mapping:
            logging.warning(f"Task {task_name} not found in task mapping")
            continue
            
        task_obj, label_list = task_mapping[task_name]
        
        # Get test data path
        test_data = task_obj.get_path()
        test_dataset = InferenceDataset(test_data, tokenizer, max_input_len, max_samples=20)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=2)
        
        all_decode_result = []
        all_labels = []
        
        with torch.no_grad():
            if label_list == []:
                # Generation-based tasks
                test_dataloader = tqdm(test_dataloader, ncols=120, leave=False, desc=f"Evaluating {task_name}")
                for test_data in test_dataloader:
                    outputs = model.generate(
                        input_ids=test_data["input_ids"].to(device),
                        attention_mask=test_data["attention_mask"].to(device),
                        max_length=max_input_len + max_output_len,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        stop_strings=["\n\n"],
                        tokenizer=tokenizer,
                    )
                    if outputs.size(1) < max_input_len + max_output_len:
                        batch_pred_padding = torch.ones((outputs.size(0), max_input_len + max_output_len - outputs.size(1)), dtype=outputs.dtype).to(device) * 2
                        outputs = torch.cat([outputs, batch_pred_padding], dim=1)
                    batch_out_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    batch_output = []
                    for idx, output in enumerate(batch_out_sentences):
                        output = output[len(test_data["input"][idx]):]
                        batch_output.append(output.strip())
                    all_decode_result.extend(batch_output)
                    all_labels.extend(test_data["label"])
            else:
                # Classification-based tasks
                test_dataloader = tqdm(test_dataloader, ncols=120, leave=False, desc=f"Evaluating {task_name}")
                for test_data in test_dataloader:
                    input_text = test_data["input"]
                    gold_label = test_data["label"]
                    new_reqs = []
                    for i in input_text:
                        for l in label_list:
                            context_enc = tokenizer.encode(i)
                            continuation_enc = tokenizer.encode(i + l)[len(context_enc):]
                            key = (i, l)
                            value = (context_enc, continuation_enc)
                            new_reqs.append((key, value))
                    input_texts = [x[0][0] + x[0][1] for x in new_reqs]
                    inputs = tokenizer(
                        input_texts,
                        return_tensors='pt',
                        padding="longest",
                        max_length=max_input_len + max_output_len,
                        truncation=True,
                        return_attention_mask=True,
                    )
                    logits = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
                    logits = F.log_softmax(logits[0], dim=-1).cpu()
                    all_result = []
                    for one_req, one_logits in zip(new_reqs, logits):
                        key, value = one_req
                        _, cont_toks = value
                        contlen = len(cont_toks)
                        one_logits = one_logits[-contlen-1 : -1].unsqueeze(0)
                        greedy_tokens = one_logits.argmax(dim=-1)
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        if len(greedy_tokens[0]) != len(cont_toks[0]):
                            answer = -float('inf')
                        else:
                            one_logits = torch.gather(one_logits, dim=2, index=cont_toks.unsqueeze(-1)).squeeze(-1)
                            answer = float(one_logits.sum())
                        all_result.append(answer)
                    all_result = np.asarray(all_result)
                    all_result = all_result.reshape(-1, len(label_list))
                    gold_label_list = [x.split(", ") for x in gold_label]
                    gold_idx = [[label_list.index(x) for x in y] for y in gold_label_list]
                    all_result = np.argmax(all_result, axis=1)
                    all_decode_result.extend(all_result)
                    all_labels.extend(gold_idx)
        
        # Compute metrics
        preds = all_decode_result
        labels = all_labels
        result = task_obj.compute_metrics(preds, labels)
        results[task_name] = result
        logging.info(f"{task_obj._cluster}_{task_obj._name}:\t{json.dumps(result)}")
    
    # Calculate weighted score
    total_score = 0
    for task_config in tasks:
        if task_config.name in results:
            task_result = results[task_config.name]
            if task_config.metric in task_result:
                total_score += task_result[task_config.metric] * task_config.weight
    
    return {"score": total_score, "results": results}


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
    batch_size: Optional[int] = None,
    task_manager: Optional[Any] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> dict:
    """
    Evaluate model using custom qu-du-tasks evaluation system
    """
    try:
        # Note: vllm parameter is ignored since we're using custom evaluation
        # The custom evaluation system uses HuggingFace transformers directly
        
        res = _eval_model(
            merged_path,
            tasks,
            model_args=model_kwargs,
            **kwargs,
        )
        return res
    finally:
        shutil.rmtree(merged_path)


evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    # monkeypatch_tqdm()
    try:
        cfg = genome.genotype_merge_config(genotype)
    except InvalidGenotypeError as e:
        logging.error("Invalid genotype", exc_info=e)
        return None
    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    run_merge(cfg, out_path=res, options=merge_options)
    return res


merge_model_ray = ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)
