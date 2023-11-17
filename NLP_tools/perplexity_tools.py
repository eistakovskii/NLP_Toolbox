# !pip install sentencepiece transformers --quiet

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np


model_name = 'cis-lmu/glot500-base'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_pppl_score(model, tokenizer, sentence) -> float:
    tensor_input = tokenizer.encode(sentence, return_tensors='pt').to(device)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1).to(device)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2].to(device)
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id).to(device)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100).to(device)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())

def get_ppl(input_str: str) -> float:
  encodings = tokenizer(input_str, return_tensors="pt")
  max_length = 512
  stride = 256
  seq_len = encodings.input_ids.size(1)

  nlls = []
  prev_end_loc = 0
  for begin_loc in range(0, seq_len, stride):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)

          # loss is calculated using CrossEntropyLoss which averages over input tokens.
          # Multiply it with trg_len to get the summation instead of average.
          # We will take average over all the tokens to get the true average
          # in the last step of this example.
          neg_log_likelihood = outputs.loss * trg_len

      nlls.append(neg_log_likelihood)

      prev_end_loc = end_loc
      if end_loc == seq_len:
          break

  ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
  # print(float(ppl))
  # print(input_str)
  return float(ppl)
