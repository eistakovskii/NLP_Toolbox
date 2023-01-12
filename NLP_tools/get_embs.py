from transformers import AutoModelForTokenClassification, AutoModel, AutoTokenizer, pipeline

import numpy as np

import torch

def embs_from_str_mlm(inp_str: str, extractor):
  '''
  Use this function to find an overage of N vectors that are produced by the extractor from the input string.

  Input:
      inp_str: input string to turn into an embedding
      extractor: pipeline of the mlm model
  Output:
      _ : returns a vector average of tokens of the input string excluding starting and trailing vectors
  '''
  temp_vecs = extractor(inp_str)[0]
  temp_vecs = temp_vecs[1:-1]
  temp_num_vecs_len = len(temp_vecs)
  temp_m_c_2 = np.zeros((temp_num_vecs_len, 768))
  for i, _ in enumerate(temp_vecs):
      temp_m_c_2[i, :] = np.array(temp_vecs[i])
  return np.average(temp_m_c_2, axis=0)
  
def embs_from_str_labse(input_str: str, model_curr, tokenizer_curr):
  '''
  Use this function to find an embedding from a whole sentence using LaBSE type of model

  Input:
      inp_str: input string to turn into an embedding
      model_curr: your model object
      tokenizer_curr: your tokenizer object
  Output:
      _ : returns a 1 by N vector representation of the input sentence
  '''
  encoded_input = tokenizer_curr(input_str, padding=True, truncation=True, max_length=64, return_tensors='pt')#.to(device)
  with torch.no_grad():
      model_output = model_curr(**encoded_input)
  embeddings = model_output.pooler_output
  embeddings = torch.nn.functional.normalize(embeddings)

  return embeddings
