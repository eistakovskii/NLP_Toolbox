## Fine-tune your target model for Named Entity Recognition

## Setup

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

Multilingual setup:

```
python ner_trainer.py \
    --batch_size 16 \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --output_dir trained_models 
```
**Step $3$: Run Inference**

Example code:
``` 
python inference.py \
    --model_name rubert \
    --model_path rubert\rubert_10000 \
```
