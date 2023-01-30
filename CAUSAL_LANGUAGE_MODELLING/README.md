## Fine-tune your target model for Causal Language Modelling

## Setup

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

``` shell
python clm_trainer.py \
    --file_path PATH_TO_YOUR_DATA.csv \
    --max_steps_num 1000 \
    --output_dir trained_models \
    --model MODEL NAME or PATH TO YOUR MODEL \
    --nickname rugpt3 \
    --say_when 2
    --learning_rate 1e-5
```
