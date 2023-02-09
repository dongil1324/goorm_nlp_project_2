
import os
import argparse
import pickle

import pandas as pd
import numpy as np
import torch
import collections
import wandb

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_metric
metric = load_metric("squad_v2")

from dataset import QADataset, QADatasetValid, QADatasetTest
from transformers import DataCollatorForTokenClassification

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', 
                    required=True,
                    help="File name to save trained model.")

    p.add_argument('--file_path',
                    required=True,
                    help="Directory where preprocessed data files located.")

    p.add_argument('--pretrained_model_name', 
                    required=True,
                    default='monologg/kobigbird-bert-base',
                    help="Set pretrained model. (Examples: klue/bert-base, monologg/kobert, ...")

    p.add_argument('--batch_size_per_device', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=2)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--n_best', type=int, default=5)
    p.add_argument('--max_answer_length', type=int, default=40)

    config = p.parse_args()

    return config

def compute_metrics(start_logits, end_logits, features, examples):  
    n_best = config.n_best
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    for i in range(len(examples)):
        example_id = examples.loc[i]["guid"]
        context = examples.loc[i]["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]    
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()   
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:      
                for end_index in end_indexes:
                  if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                  if (end_index < start_index
                        or end_index - start_index + 1 > config.max_answer_length):
                    continue

                  answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],   
                        "logit_score": start_logit[start_index] + end_logit[end_index]}
                  answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
          best_answer = sorted(answers, key=lambda x: x["logit_score"], reverse=True)[0]
        else:
          best_answer = {"text": "", "score": 0.0}

        answer = best_answer["text"]
        predictions[examples.loc[i]["guid"]] = answer  
    
    predicted_answers = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
    theoretical_answers = []
    for i in range(len(examples)):
      if len(examples.loc[i]["text"]) == 0:
        theoretical_answers.append({"id": examples.loc[i]["guid"], "answers": {'text': [""], 'answer_start': examples.loc[i]['answer_start']}})
      else:
        theoretical_answers.append({"id": examples.loc[i]["guid"], "answers": {'text': examples.loc[i]["text"], 'answer_start': examples.loc[i]['answer_start']}})
    
    result = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    return {
        "exact": result['exact'],
        "f1": result['f1']}


def main(config):

    # read preprocessed data files
    with open(os.path.join(config.file_path, 'preprocessed_train.pickle'), 'rb') as fr:
      train = pickle.load(fr)
    with open(os.path.join(config.file_path, 'preprocessed_validation.pickle'), 'rb') as fr:
      valid = pickle.load(fr)
    validation = pd.read_pickle(os.path.join(config.file_path, 'validation.pkl'))
    
    train_dataset = QADataset(train['input_ids'], train['token_type_ids'], train['attention_mask'], train['start_positions'], train['end_positions'])
    validation_dataset = QADatasetValid(valid['input_ids'], valid['token_type_ids'], valid['attention_mask'], valid['offset_mapping'], valid['example_id'])
    print(len(train_dataset), len(validation_dataset))

    total_batch_size = config.batch_size_per_device
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print('#total_iters =', n_total_iterations, '#warmup_iters =', n_warmup_steps)
    
    # import tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(config.pretrained_model_name)
    wandb.login()
    wandb.init(project = 'dongil_test_MRC', entity = 'goorm_nlp_project_1', reinit=True)
    wandb.run.name=config.model_fn
    # fine-tuning using huggingface trainer
    training_args = TrainingArguments(
        output_dir='./data/checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        report_to = 'wandb'
        #run_name = config.model_fn
        )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
                model=model,
                args=training_args, 
                train_dataset=train_dataset,  
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer)

    trainer.train()
    
    torch.save(model.state_dict(), config.model_fn)

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    print(compute_metrics(start_logits, end_logits, validation_dataset, validation))

if __name__ == "__main__":
    config = define_argparser()
    main(config)