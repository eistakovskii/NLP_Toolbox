import numpy as np
import torch
from tqdm.auto import tqdm

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
  
def get_labse_sentence_emb(input_str: str, model_curr, tokenizer_curr):
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

def get_cls_token(str_in: str, model_curr, tokenizer_curr):
    """
    This function exploits the NSP (next sentece prediction) task with which the Bert was pretrained along the MLM task
    and uses [CLS] token to encode the whole sentence.
    Note that this function assumes that your using GPU
    The function returns a single vector for an input string with dimensions (, 768)

    Input:
        str_in: your input string
        model_curr: bert model to vectorize the strings
        tokenizer_curr: model's tokenizer
    Output:
        cls_token: the function returns a numpy vector with the following dimensions: (, 768) 
    """
    
    input_ids = torch.tensor(tokenizer_curr.encode(str_in, truncation=True, max_length=512)).unsqueeze(0).to('cuda') 
    outputs = model_curr(input_ids) # Forward pass with the input
    last_hidden_states = outputs[0]
    cls_token = last_hidden_states[0][0] # Extract the CLS token and discard everything else
    cls_token = cls_token.detach().cpu().numpy()
    
    return cls_token

def embs_from_str_mlm(data_in: list, model_curr, tokenizer_curr):
  """
  This function exploits the NSP (next sentece prediction) task with which the Bert was pretrained along the MLM task
  and uses [CLS] token to encode the whole sentence.
  Note that this function assumes that your using GPU
  
  Input:
      data_in: your data in the form of list of strings
      model_curr: bert model to vectorize the strings
      tokenizer_curr: model's tokenizer
  Output:
      out_matrix: as the name suggests the function returns a numpy matrix with the following dimensions: (number of data samples, 768) 
  """
  data_len = len(data_in)
  out_matrix = np.zeros((data_len, 768)) # Initiate with zeros a matrix with dimensions (number of train sample, 768)
  
  for ind, val in tqdm(enumerate(data_in)):
      input_ids = torch.tensor(tokenizer_curr.encode(val)).unsqueeze(0).to('cuda') 
      outputs = model_curr(input_ids) # Forward pass with the input
      last_hidden_states = outputs[0]
      cls_token = last_hidden_states[0][0] # Extract the CLS token and discard everything else
      out_matrix[ind, :] = cls_token.detach().cpu().numpy() # Insert the cls token in the matrix intitiated above
  
  return out_matrix

