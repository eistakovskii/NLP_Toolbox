# Ref: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py

import datasets
import os
from pathlib import Path
from datasets import ClassLabel, DownloadConfig

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
    """NER_dataset dataset."""

    BUILDER_CONFIGS = [
        NER_dataset_Config(name="NER_dataset", version=datasets.Version("1.0.0"), description="NER_dataset dataset"),
    ]

    def __init__(self,
                 *args,
                 cache_dir,
                 main_dir_path,
                 train_file="train.txt",
                 val_file="valid.txt",
                 test_file="test.txt",
                 ner_tags,
                 **kwargs):
        self._ner_tags = ner_tags
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file
        self._main_dir_path = main_dir_path
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
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self._main_dir_path + self._train_file}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self._main_dir_path + self._val_file}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self._main_dir_path + self._test_file}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
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

    def __init__(self, mp, tg):
        self._mp = mp
        self._tg = tg
        cache_dir = os.path.join(str(Path.home()), '.hf_ner_dataset')
        print("Cache directory: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = DownloadConfig(cache_dir=cache_dir)
        self._dataset = NER_dataset(cache_dir=cache_dir, main_dir_path = self._mp, ner_tags = self._tg)
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


if __name__ == '__main__':
    print('\nEnter the path to your data:')
    main_path_in = input()
    tg_in = ("B-NAVY", "I-NAVY", "B-ARMY", "I-ARMY", "B-AIR_FORCE", "I-AIR_FORCE", "B-MISSILES", "I-MISSILES", "O")
    dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in).dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


    print("First sample: ", dataset['train'][0])