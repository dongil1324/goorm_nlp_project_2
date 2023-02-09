import torch
from torch.utils.data import Dataset


class QADataset(Dataset): 

    def __init__(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids 
        self.attention_mask = attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):              
        input_id = self.input_ids[item]
        token_type_id = self.token_type_ids[item]
        attention_mask = self.attention_mask[item]
        start_position = self.start_positions[item]   
        end_position = self.end_positions[item]       

        return {
            'input_ids': input_id,
            'token_type_ids' : token_type_id,             
            'attention_mask' : attention_mask,  
            'start_positions': start_position,
            'end_positions' : end_position
        }


class QADatasetValid(Dataset): 

    def __init__(self, input_ids, token_type_ids, attention_mask, offset_mapping, example_id):
      self.input_ids = input_ids
      self.token_type_ids = token_type_ids 
      self.attention_mask = attention_mask
      self.offset_mapping = offset_mapping
      self.example_id = example_id

    def __len__(self):
      return len(self.input_ids)

    def __getitem__(self, item):
      input_id = self.input_ids[item]
      token_type_id = self.token_type_ids[item]
      attention_mask = self.attention_mask[item]
      offset_mapping = self.offset_mapping[item]   
      example_id = self.example_id[item]
      
      result = {'input_ids': input_id,
            'token_type_ids' : token_type_id,             
            'attention_mask' : attention_mask,  
            'offset_mapping': offset_mapping,
            'example_id' : example_id}
      return result


class QADatasetTest(Dataset): 

    def __init__(self, input_ids, token_type_ids, attention_mask):
      self.input_ids = input_ids
      self.token_type_ids = token_type_ids 
      self.attention_mask = attention_mask

    def __len__(self):
      return len(self.input_ids)

    def __getitem__(self, item):
      input_id = self.input_ids[item]
      token_type_id = self.token_type_ids[item]
      attention_mask = self.attention_mask[item]


      result = {'input_ids': input_id,
            'token_type_ids' : token_type_id,             
            'attention_mask' : attention_mask}
      return result