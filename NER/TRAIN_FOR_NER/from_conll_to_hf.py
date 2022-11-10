# Ref: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py

import datasets
import os
from pathlib import Path
from datasets import ClassLabel, DownloadConfig

from split_data import do_the_splits

import argparse

logger = datasets.logging.get_logger(__name__)


_CITATION = ""
_DESCRIPTION = """"""



class NER_dataset_Config(datasets.BuilderConfig):
    """BuilderConfig for NER_dataset"""

    def __init__(self, **kwargs):
        """BuilderConfig for NER_dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NER_dataset_Config, self).__init__(**kwargs)


class NER_dataset(datasets.GeneratorBasedBuilder):
    """NER_dataset."""

    BUILDER_CONFIGS = [
        NER_dataset_Config(name="NER_dataset", version=datasets.Version("1.0.0"), description="NER_dataset"),
    ]

    def __init__(self,
                 *args,
                 cache_dir,
                 train_list,
                 val_list,
                 test_list,
                 ner_tags,
                 **kwargs):
        self._ner_tags = ner_tags
        self._train_list = train_list
        self._val_list = val_list
        self._test_list = test_list
        super(NER_dataset, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.
        
        """

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"curr_split": self._train_list}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"curr_split": self._val_list}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"curr_split": self._test_list}),
        ]

    def _generate_examples(self, curr_split):

        """
        Process each split
        """
        
        logger.info("⏳ Generating examples from = %s", curr_split)
        
        guid = 0
        tokens = []
        ner_tags = []
        
        for line in curr_split:
            if line == "" or line == "\n":
                if tokens:
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    guid += 1
                    tokens = []
                    ner_tags = []
            else:
                # NER_dataset tokens are space separated
                splits = line.split(" ")
                tokens.append(splits[0])
                ner_tags.append(splits[1].rstrip())
        # last example
        yield guid, {
            "id": str(guid),
            "tokens": tokens,
            "ner_tags": ner_tags,
            }


class HF_NER_dataset(object):
    """
    Convert into a Hugging Face type dataset
    """
    NAME = "HF_NER_dataset"

    def __init__(self, mp, tg, exp_bool=0):
        
        self._mp = mp
        self._tg = tg
        self._export_bool = exp_bool
        
        train_split_get, val_split_get, test_split_get, ner_tags_get = do_the_splits(self._mp, self._tg, self._export_bool)
        ner_tags_get = tuple(ner_tags_get)
        
        cache_dir = os.path.join(str(Path.home()), '.hf_ner_dataset')
        print("Cache directory: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = DownloadConfig(cache_dir=cache_dir)
        
        self._dataset = NER_dataset(cache_dir=cache_dir, 
            train_list = train_split_get,
            val_list= val_split_get,
            test_list= test_split_get,
            ner_tags = ner_tags_get)
        
        print("Cache1 directory: ",  self._dataset.cache_dir)
        
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def test(self):
        return self._dataset["test"]

    def validation(self):
        return self._dataset["validation"]

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
    
#     parser.add_argument(
#         "--file_path", type=str, help="path to your conll file", required=True
#     )
#     parser.add_argument(
#         "--tags", type=str, help="your NE tags", required=True
#     )
#     parser.add_argument(
#         "--export", type=bool, default=False, help="export the splits locally"
#     )
    
#     args = parser.parse_args()
    
#     main_path_in = args.file_path

#     tg_in_t = args.tags
    
#     tg_in = tg_in_t.split(',')

#     exp_bool_in = args.export

#     dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in, exp_bool=exp_bool_in).dataset

#     print(dataset['train'])
#     print(dataset['test'])
#     print(dataset['validation'])

#     print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


#     print("First sample: ", dataset['train'][0])
