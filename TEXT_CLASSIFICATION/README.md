## Fine-tune your target model for (Binary) Text Classification 

## Training

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

``` shell
python classification_trainer.py \ # or classification_adapter_trainer.py
    --file_path PATH_TO_YOUR_DATA.csv \
    --max_epoch_num 10 \
    --output_dir trained_models \
    --model MODEL NAME or PATH TO YOUR MODEL \
    --nickname YOUR_MODEL \ # e.g. bert-base-multilingual-cased
    --say_when 5 \
    --zero_means WHAT_0_STANDS_FOR \ # e.g. Negative
    --one_means WHAT_1_STANDS_FOR # e.g. Positive
```
## Inference

**To run the fully fine-tuned model in inference mode**

``` python
from transformers import pipeline

classifier = pipeline("text-classification", model = "PATH_TO_YOUR_MODEL or HF_MODEL_NAME")

classifier("YOUR_TEXT_TO_CLASSIFY")
```

**To run the adapter model in inference mode**

``` python
from transformers import AutoTokenizer, AutoAdapterModel, pipeline

tokenizer = AutoTokenizer.from_pretrained('YOUR_MAIN_MODEL')
model = AutoAdapterModel.from_pretrained('YOUR_MAIN_MODEL')

adapter_name = model.load_adapter(r"PATH_TO_YOUR_ADAPTER")
model.load_head(r"PATH_TO_YOUR_ADAPTER")

model.set_active_adapters(adapter_name)

classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)

classifier("YOUR_TEXT_TO_CLASSIFY")
```

