import os
import csv
import argparse
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np
import collections

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from dataset import QADataset, QADatasetValid, QADatasetTest


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--file_path', required=True)
    p.add_argument('--pretrained_model_name', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_best', type=int, default=5)
    p.add_argument('--max_answer_length', type=int, default=40)
    p.add_argument('--output_name', default = 'submission.csv')
    config = p.parse_args()

    return config


def main(config):
    with open(os.path.join(config.file_path, 'preprocessed_validation.pickle'), 'rb') as fr:
      preprocessed_test = pickle.load(fr)
    test = pd.read_pickle(os.path.join(config.file_path, 'validation.pkl'))

    test_dataset = QADatasetValid(preprocessed_test['input_ids'], preprocessed_test['token_type_ids'], preprocessed_test['attention_mask'], preprocessed_test['offset_mapping'], preprocessed_test['example_id'])
    test_set = QADatasetTest(preprocessed_test['input_ids'], preprocessed_test['token_type_ids'], preprocessed_test['attention_mask'])

    saved_data = torch.load(
        config.model_fn,
        map_location='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(config.pretrained_model_name)
        model.load_state_dict(saved_data) 

        if torch.cuda.is_available():
            model.cuda()
        device = next(model.parameters()).device

        test_dataloader = DataLoader(test_set, collate_fn=default_data_collator, batch_size=config.batch_size, shuffle=False)

        # Don't forget turn-on evaluation mode.
        model.eval()

        # Predictions
        start_logits = []
        end_logits = []
        for batch in tqdm(test_dataloader):
            x = torch.tensor(batch['input_ids']).to(device)
            token_type_ids = torch.tensor(batch['token_type_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            outputs = model(x,  token_type_ids=token_type_ids, attention_mask=attention_mask)
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())
            
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(test_dataset)]
        end_logits = end_logits[: len(test_dataset)]
    
    # create answers
    n_best = config.n_best

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(test_dataset):
        example_to_features[feature["example_id"]].append(idx)  

    predicted_answers = []
    for i in range(len(test)):
        example_id = test.loc[i]["guid"]
        context = test.loc[i]["context"]
        answers = []
    
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]   
            end_logit = end_logits[feature_index]
            offsets = test_dataset[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()   
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index
                        or end_index - start_index + 1 > config.max_answer_length):
                        continue

                    answer = {"text": context[offsets[start_index][0] : offsets[end_index][1]],   
                            "logit_score": start_logit[start_index] + end_logit[end_index]}
                    answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""}) 

    os.makedirs('out', exist_ok=True)
    with open(f'out/{config.output_name}', 'w') as fd:
      writer = csv.writer(fd)
      writer.writerow(['Id', 'Predicted'])  
      rows = []       
      for i in range(len(predicted_answers)):
        pred_guid, pred_ans = predicted_answers[i]['id'], predicted_answers[i]['prediction_text']
        rows.append([pred_guid, pred_ans])
      writer.writerows(rows)

if __name__ == '__main__':
    config = define_argparser()
    main(config)