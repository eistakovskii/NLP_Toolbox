## Fine-tune your target model for Masked Language Modelling

## Training

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

``` shell
python mlm_trainer.py \
    --file_path PATH_TO_YOUR_DATA.csv \
    --max_steps_num 10000 \
    --output_dir trained_models \
    --model MODEL NAME or PATH TO YOUR MODEL \
    --nickname bert \
    --say_when 5
```
