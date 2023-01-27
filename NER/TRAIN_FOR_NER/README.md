## Fine-tune your target model for Named Entity Recognition

## Training

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Training**

``` shell
python ner_trainer.py \
    --file_path PATH_TO_YOUR_DATA.txt \
    --tags LOC,ORG,MISC \
    --batch_size 16 \
    --max_steps 1000 \
    --learning_rate 2e-5 \
    --output_dir trained_models \
    --model MODEL NAME or PATH TO YOUR MODEL \
    --nickname bert
```
## Inference
To run the trained model in inference:

``` shell
python inference_ner.py \
    --checkpoint_path PATH_TO_YOUR_MODEL \
    --tags LOC,ORG,MISC \ # Your tags separated with commas and w/o spaces
    --input_text \ # Input here your text
```
or in a notebook:
``` python
from inference_ner import inference_ner_with_tags

inference_ner_with_tags('/PATH_TOU_YOUR_MODEL/checkpoint-X/', 
                        ['LOC','ORG','MISC'], 
                        'YOUR TEXT TO TAG')
```                        
