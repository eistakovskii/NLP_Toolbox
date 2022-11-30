## Fine-tune your target model for Text Classification

## Setup

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

``` shell
python classification_trainer.py \
    --file_path PATH_TO_YOUR_DATA.csv \
    --max_epoch_num 10 \
    --output_dir trained_models \
    --model MODEL NAME or PATH TO YOUR MODEL \
    --nickname bert \
    --say_when 5
```
