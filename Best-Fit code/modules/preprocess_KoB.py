import os
import argparse
from tqdm import tqdm
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--data_path',
        required=True,
        help="Directory where data files located.")

    p.add_argument(
        "--save_path",
        required=True,
        help="Directory to save preprocessed dataset.")

    p.add_argument(
        '--test_size',
        default=.2,
        type=float,
        help="Set test size. Input float number")

    p.add_argument('--pretrained_model_name', 
                    required=True,
                    default='monologg/kobigbird-bert-base',
                    help="Set pretrained model. (Examples: klue/bert-base, monologg/kobert, ...")
    
    p.add_argument('--max_length', type=int, default=384)
    p.add_argument('--stride', type=int, default=50)
    p.add_argument('--train_fn', 
                    default='train.json',
                    help='traindata filename for train')

    config = p.parse_args()
    return config


def preprocess_training_examples(samples):
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    
    inputs = tokenizer(
        samples["question"].tolist(),
        samples["context"].tolist(),
        max_length = config.max_length,
        stride=50,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding = "max_length"
    )
   
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answer_starts = samples['answer_start']
    texts = samples['text']
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)
        sample_idx = sample_map[i]
        answer_start = answer_starts[sample_idx]
        text = texts[sample_idx]

        if len(answer_start) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

        else:
            start_char = answer_start[0]
            end_char = answer_start[0] + len(text[0])
        
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# preprocess_validation_example function
def preprocess_validation_examples(examples):
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    inputs = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        max_length=config.max_length,
        truncation="only_second",
        stride=config.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["guid"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [o if k==0 or sequence_ids[k] == 1 else None for k, o in enumerate(offset)]  

    inputs["example_id"] = example_ids
    return inputs


def main(config):
    datapath = config.data_path
    savepath = config.save_path
    train_fn = config.train_fn
   

    # read data files into dataframe
    train = pd.read_json(os.path.join(datapath, train_fn))
    test = pd.read_json(os.path.join(datapath, 'test.json'))

    # train data
    cols = ['context', 'guid', 'question', 'answer_start', 'text']

    comp_list = []
    for row in train['data']:
        for i in range(len(row['paragraphs'])):
            if len(row['paragraphs'][i]['qas'][0]['answers']) !=0:
                temp_list = []
                temp_list.append(row['paragraphs'][i]['context'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['guid'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['question'])
                temp_list.append([row['paragraphs'][i]['qas'][0]['answers'][0]['answer_start']])
                temp_list.append([row['paragraphs'][i]['qas'][0]['answers'][0]['text']])
                comp_list.append(temp_list)

            else:
                temp_list = []
                temp_list.append(row['paragraphs'][i]['context'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['guid'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['question'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['answers'])
                temp_list.append(row['paragraphs'][i]['qas'][0]['answers'])
                comp_list.append(temp_list)

    train = pd.DataFrame(comp_list, columns=cols) 

    # test data
    cols = ['context', 'guid', 'question']

    comp_list = []
    for row in test['data']:
        for i in range(len(row['paragraphs'])):
            for j in range(len(row['paragraphs'][i]['qas'])):
                temp_list = []
                temp_list.append(row['paragraphs'][i]['context'])
                temp_list.append(row['paragraphs'][i]['qas'][j]['guid'])
                temp_list.append(row['paragraphs'][i]['qas'][j]['question'])
                comp_list.append(temp_list)

    test = pd.DataFrame(comp_list, columns=cols) 

    # answer_start correction
    text_comparision = []
    for i in range(len(train['context'])):
        if len(train['answer_start'][i]) !=0:
            text_comparision.append([train['context'][i][train['answer_start'][i][0]: train['answer_start'][i][0]+len(train['text'][i][0])]])
        else:
            text_comparision.append(train['text'][i])

    train['text_comparision'] = text_comparision

    answer_start = []
    for i in range(len(train['answer_start'])):
        if train['text'][i]!= train['text_comparision'][i]:
            answer_start.append([train['answer_start'][i][0]+1])
        else:
            answer_start.append(train['answer_start'][i])

    train['answer_start'] = answer_start
    train = train.drop(['text_comparision'], axis=1)
    print('Train shape', train.shape)

    # train and validation split
    train, validation = train_test_split(train, test_size=config.test_size, random_state=42, shuffle=True)
    train = train.reset_index()
    validation = validation.reset_index(drop=True)
    print('train length: {}, validation length: {}'.format(len(train), len(validation)))

    # preprocess data
    preprocessed_train = preprocess_training_examples(train[0::])
    preprocessed_validation = preprocess_validation_examples(validation[0::])
    preprocessed_test = preprocess_validation_examples(test[0::])
 
    # save files
    with open(os.path.join(savepath, 'preprocessed_train.pickle'),'wb') as fw:
        pickle.dump(preprocessed_train, fw)

    with open(os.path.join(savepath, 'preprocessed_validation.pickle'),'wb') as fw:
        pickle.dump(preprocessed_validation, fw)

    with open(os.path.join(savepath, 'preprocessed_test.pickle'),'wb') as fw:
        pickle.dump(preprocessed_test, fw)

    validation = validation.to_pickle(os.path.join(savepath, 'validation.pkl'))
    test = test.to_pickle(os.path.join(savepath, 'test.pkl'))
    print('finished preprocessing')


if __name__ == "__main__":
    config = define_argparser()
    main(config)