## Prepare an artificial CoLa dataset for a target language

## Training

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Install the corrupter library**

```
pip install git+https://github.com/eistakovskii/text_corruption_plus.git@master
```

**Step $3$: Run the script in the notebook**

``` python
get_dataset('de', 50000)
get_dataset('fr', 50000)
```
