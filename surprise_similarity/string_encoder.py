import numpy as np
import math
import torch
import json
import os
import time
from collections import OrderedDict
import pickle
import _pickle as cPickle
import hashlib
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from torch import nn, Tensor
from typing import Iterable, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROGRESS_THERE = False


def print_progress(percent, label):
    global PROGRESS_THERE
    if PROGRESS_THERE:
        print(f"\r{label} {str(percent)}%", end="")
    else:
        PROGRESS_THERE = True
        print(f"{label} {str(percent)}%", end="")

def end_progress():
    global PROGRESS_THERE
    PROGRESS_THERE = False
    print("")

def cache_hash(input):
    if not isinstance(input, bytes):
        try:
            bytes_ = json.dumps(input).encode()
        except Exception:
            bytes_ = cPickle.dumps(input, protocol=4)
    else:
        bytes_ = input
    ret = int.from_bytes(hashlib.sha256(bytes_).digest(), byteorder='big')
    shift = 256 - ret.bit_length()
    return ret << shift >> 130


class LRUCache:
    def __init__(self, capacity: int, copy=False):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.copy = copy

    def get(self, key: int, default_value="NoValue") -> int:
        if key not in self.cache:
            return default_value
        else:
            self.cache.move_to_end(key)
            if self.copy:
                return pickle.loads(pickle.dumps(self.cache[key]))
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

def cosine_similarity(a, b):
    unity = torch.Tensor([1.0]).to(b.device).type(a.type())
    if not torch.isclose(a[0].norm(), unity):
        a = (a.T / a.norm(dim=1)).T
    if not torch.isclose(b[0].norm(), unity):
        b = (b.T / b.norm(dim=1)).T
    return torch.matmul(a, b.T)


class xEntropyCosSimilarityLoss(nn.Module):

    """
    xEntropyCosSimilarityLoss is built over the CosineSimilarityLoss but implements a cross-entropy loss function
    over the batch. It does by taking InputExamples consisting of two texts and a float label.
    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    It applies a ReLU to make sure the output is positive and it adds a +1e-10 offset to avoid instability of the logarithm:

    prob = ReLU(cosine_sim(u,v))+1e-10

    It then minimizes the following loss: -(input_label*log(prob)*(1-prob)**gamma+(1-input_label)*log(1-prob)*(prob)**gamma).

    The factor (1-prob)**gamma and (prob)**gamma reduce the relative loss for well-classified examples thus focusing on the
    hard, missclassified examples (this is also called a focal loss). The model also stores the value of the loss with gamma=0
    (called tot_no_gamma) which is averaged over the entire dataset and used by the train methods to define when to stop.

    Args:
        model: SentenceTranformer model
        gamma: float, default=0
        len_data: int, default=1
            Number of batches in the dataset (e.g. len of the dataloader)

    Example::
            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
            model = SentenceTransformer("sentence-transformers/sentence-t5-base")
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
            train_loss = xEntropyCosSimilarityLoss(model,len_data=len(train_dataloader),gamma=1)

    """

    def __init__(self, model: SentenceTransformer, gamma=1, len_data=1):
        super(xEntropyCosSimilarityLoss, self).__init__()
        self.model = model
        self.gamma = gamma
        self.len_data = len_data
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.iter = 0
        self.tot_no_gamma = []

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if self.iter % self.len_data == 0:
            self.tot_no_gamma = []
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        prob = torch.cosine_similarity(embeddings[0], embeddings[1])
        m = torch.nn.ReLU()
        prob_1 = m(prob).to(self.device) + 1e-10
        prob_0 = 1 - prob_1
        output_no_gamma = -labels.view(-1) * torch.log(prob_1) - (1 - labels.view(-1)) * torch.log(prob_0)
        output_gamma = (
            -labels.view(-1) * torch.log(prob_1) * (1 - prob_1) ** self.gamma
            - (1 - labels.view(-1)) * torch.log(prob_0) * (1 - prob_0) ** self.gamma
        )
        self.tot_no_gamma.append(torch.sum(output_no_gamma) / len(output_no_gamma))
        self.iter += 1
        return torch.sum(output_gamma) / len(output_gamma)


class StringEncoder:
    """
    The StringEncoder is a wrapper around a SentenceTransformer model that can be used to encode strings
    and train the underlying model on a given dataset.

    Args:
        model_name: str, default="sentence-transformers/sentence-t5-base"
            The name of the SentenceTransformer model to use
        cache_size: int, default=33000000
            The size of the cache used to store the encoded strings

    Main methods:
        - encode: encode a list of strings
        - train: train the underlying SentenceTransformer model on a given dataset

    """
    def __init__(self,
                model_name="sentence-transformers/sentence-t5-base",
                cache_size=3000000,) -> None:
        self.string_encode_cache = LRUCache(capacity=cache_size)
        self.cache_size = cache_size
        self.model_name = model_name
        self.transformers_tokenizers = None
        self.model = None
        self.device = None
        self.trained = False
        self.max_itns = False

    def to(self, device):
        if self.model is not None:
            self.device = device
            self.model.to(device)
        return self

    @property
    def cache_size(self):
        return self.string_encode_cache.capacity

    @cache_size.setter
    def cache_size(self, cache_size):
        self.string_encode_cache.capacity = cache_size

    def _set_sentence_transformers_model(self, model_name):
        if self.model_name == model_name and self.model is not None:
            return None
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.transformers_tokenizers = None
        self.model_name = model_name

    def _set_transformers_model(self, model_name):
        if self.model_name == model_name and self.model is not None:
            return None
        from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
        import transformers
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        transformers.logging.set_verbosity_error()
        if "flan-t5" in self.model_name:
            if self.model is None:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model = self.model.to(device)
                self.transformers_tokenizers = AutoTokenizer.from_pretrained(model_name)
                self.model_name = model_name
            elif self.model_name != model_name:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model = self.model.to(device)
                self.transformers_tokenizers = AutoTokenizer.from_pretrained(model_name)
                self.model_name = model_name
        else:
            if self.model is None:
                self.model = AutoModel.from_pretrained(model_name)
                self.model = self.model.to(device)
                self.transformers_tokenizers = AutoTokenizer.from_pretrained(model_name)
                self.model_name = model_name
            elif self.model_name != model_name:
                self.model = AutoModel.from_pretrained(model_name)
                self.model = self.model.to(device)
                self.transformers_tokenizers = AutoTokenizer.from_pretrained(model_name)
                self.model_name = model_name

    def _transformers_encode(self, string, model_name=None, batch_size=32, show_progress_bar=False):
        from torch.nn import functional as F
        import transformers
        import torch

        if model_name is None:
            model_name = self.model_name

        self._set_transformers_model(model_name)
        if not isinstance(string, list):
            use_string = [string]
        else:
            use_string = string
        all_embs = []
        n_batches = math.ceil(len(use_string) / batch_size)
        if show_progress_bar:
            from tqdm import trange

            iterator = trange(n_batches, desc=model_name)
        else:
            iterator = range(n_batches)
        for i in iterator:
            if len(use_string[i * batch_size : (i + 1) * batch_size]) == 0:
                continue
            inputs = self.transformers_tokenizers.batch_encode_plus(
                use_string[i * batch_size : (i + 1) * batch_size], return_tensors="pt", padding="longest"
            )
            input_ids = inputs["input_ids"]
            input_ids = input_ids.to(self.device)
            attention_mask = inputs["attention_mask"]
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "encoder"):
                    output = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
                else:
                    output = self.model(input_ids, attention_mask=attention_mask)[0]
            flattened = attention_mask.view(-1)
            flattened = flattened.float()
            masked = (output.view(-1, output.shape[-1]) * flattened.view(-1, 1)).view(
                output.shape[0], output.shape[1], output.shape[2]
            )
            pooled_embeddings = masked.sum(dim=1) / attention_mask.sum(dim=1).view(-1, 1)
            pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, dim=1).to("cpu").numpy()
            all_embs.append(pooled_embeddings)
        all_embs = np.concatenate(all_embs, axis=0)
        if isinstance(string, list):
            return all_embs
        return all_embs[0] 

    def get_state_dict(self, model):
        if isinstance(model, torch.nn.DataParallel):
            best_state_dict = model.module.state_dict()
        else:
            best_state_dict = model.state_dict()
        return best_state_dict

    def set_model(self, model_name):
        try:
            self._set_sentence_transformers_model(model_name)
        except:
            self._set_transformers_model(model_name)

    @staticmethod
    def check_binary_negative(queries):
        if len(set(queries)) == 1:
            return True
        elif set(queries) == set(queries + [None]):
            return True
        return False

    @staticmethod
    def multilabel_scores(targets, predictions, zero_division=1):
        """
        Compute the precision, recall, and F1 score for a multi-label non-exclusive classification problem.

        Args:
        - targets (list of lists): the true labels for each example
        - predictions (list of lists): the predicted labels for each example

        Returns:
        - precision (float): the precision score
        - recall (float): the recall score
        - f1 (float): the F1 score
        """
        targets = [set(it) if not isinstance(it, str) else set(it.split()) for it in targets]
        predictions = [set(it) if not isinstance(it, str) else set(it.split()) for it in predictions]
        # Initialize the true positive, false negative, and false positive counts to zero
        tp = 0
        fn = 0
        fp = 0

        # Loop over each example
        for i in range(len(targets)):
            # Compute the set of true labels and predicted labels for this example
            true_labels = targets[i]
            pred_labels = predictions[i]

            # Compute the true positive, false negative, and false positive counts for this example
            tp += len(true_labels.intersection(pred_labels))
            fn += len(true_labels.difference(pred_labels))
            fp += len(pred_labels.difference(true_labels))

        # Compute the precision, recall, and F1 score
        if tp + fp == 0:
            precision = zero_division
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = zero_division
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = zero_division
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def train(
        self,
        keys,
        matching_queries,
        shuffle=True,
        target_train_f1=None,
        target_train_loss=0.3,
        show_progress=False,
        max_its=50,
        labels=None,
        min_its=1,
        lr_factor=1,
        loss="xCos",
        gamma=1,
        weight_decay=0.01,
        exclusive=False,
        non_exclusive_threshold=0.5,
    ):
        # labels in default format: list of strings
        if labels is not None:
            labels = list(set([it for it in labels if it is not None]))
            if len(labels) == 1:
                exclusive = False
        # cast input in standard format:keys are uniqye, matching_queries: list, exclusive = True/False
        assert len(keys) == len(matching_queries), "keys and matching_queries must have the same length"
        key_dict = {}
        for i in range(len(matching_queries)):
            if keys[i] not in key_dict:
                key_dict[keys[i]] = set()
            if isinstance(matching_queries[i], list):
                exclusive = False
                key_dict[keys[i]] = key_dict[keys[i]].union(set(matching_queries[i]))
            elif matching_queries[i] is None:
                pass
            else:
                key_dict[keys[i]].add(matching_queries[i])
        keys = []
        matching_queries = []
        all_matching_queries = set()
        for k, v in key_dict.items():
            keys.append(k)
            matching_queries.append(v)
            all_matching_queries = all_matching_queries.union(v)
        if labels is None:
            labels = list(all_matching_queries)
            if len(labels) == 1:
                exclusive = False

        self.gamma = gamma
        if self.model is None:
            self.set_model(self.model_name)
        st = time.time()
        assert self.model_name.startswith(
            "sentence-transformers"
        ), "Fine tuning is only supported on sentence-transformers models"
        train_examples = []
        training_data = list(zip(keys, matching_queries))

        for i, itm in enumerate(zip(keys, matching_queries)):
            text, label = itm
            for query in labels:
                if query in label:
                    train_examples.append(InputExample(texts=[text, query], label=1.0))
                else:
                    train_examples.append(InputExample(texts=[text, query], label=0.05 if len(label) > 0 else 0.0))

        train_dataloader = DataLoader(train_examples, shuffle=shuffle, batch_size=64)
        if loss == "xCos":
            train_loss = xEntropyCosSimilarityLoss(self.model, len_data=len(train_dataloader), gamma=self.gamma)
        elif loss == "CosS":
            train_loss = losses.CosineSimilarityLoss(self.model)
        i = 0
        trained = False
        label_dict = {i: labels[i] for i in range(len(labels))}
        print(f"Training on {len(train_examples)} examples...")
        last_progress = 0
        num_train_it = 0
        current_train_f1 = 0
        current_train_loss = 999
        start_train_f1 = 0
        while (
            (current_train_f1 < target_train_f1 if target_train_f1 is not None else False)
            or (current_train_loss > target_train_loss if target_train_loss is not None else False)
            or i < min_its
        ):
            if i > max_its:
                print(
                    f"\nReached max iterations ({max_its}) to F1 of {current_train_f1} without reaching the target train ({target_train_f1})."
                )
                print(zip(keys, matching_queries))
                self.max_itns = True
                self.tot_itns = i
                break
            if show_progress and i == 0:
                print_progress(0, f"Training {self.model_name.split('/')[-1]}:")
            # Idea: implement training data creation to assign 0 to all labels that have a higher cosine similarity than the target only
            if i > min_its - 1:
                key_embs = self.encode_string([it[0] for it in training_data])
                query_embs = self.encode_string(labels)
                result = torch.matmul(key_embs, query_embs.T).numpy()
                if exclusive:
                    contrastive_surprise_preds = [label_dict[itm] for itm in np.argmax(result, axis=1)]
                else:
                    indices = []
                    for row in result:
                        indices.append(list(np.where(row > non_exclusive_threshold)[0]))
                    contrastive_surprise_preds = [set(label_dict[it] for it in itm) for itm in indices]
                current_train_f1 = self.multilabel_scores(
                    [it[1] for it in training_data],
                    contrastive_surprise_preds,
                    zero_division=1,
                )["f1"]                
                targets=[it[1] for it in training_data]
                preds=contrastive_surprise_preds
                fraction_correct=sum([targets[i]== preds[i] for i in range(len(targets))])/len(targets)
                if show_progress:
                    print('fraction correct:', fraction_correct)
                    print('current_train_f1:', current_train_f1)
                current_train_f1=fraction_correct
                number_batches=len(train_dataloader)
                current_train_loss = sum(train_loss.tot_no_gamma)/number_batches
                if i == 0:
                    start_train_f1 = current_train_f1

            if (
                (current_train_f1 < target_train_f1 if target_train_f1 is not None else False)
                or (current_train_loss > target_train_loss if target_train_loss is not None else False)
                or i < min_its
            ):
                num_train_it += 1
                self.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=3,
                    warmup_steps=0,
                    weight_decay=weight_decay,
                    optimizer_params={"lr": lr_factor * 2e-5},
                    show_progress_bar=False,
                )
                self.string_encode_cache = LRUCache(capacity=self.cache_size)
            if show_progress:
                if target_train_f1 - start_train_f1 <= 0:
                    percent = 100
                else:
                    percent = min(
                        [
                            100,
                            round(
                                float((current_train_f1 - start_train_f1 if current_train_f1 - start_train_f1>0 else 0) / (target_train_f1 - start_train_f1)) ** 2.8
                                * 100
                            )
                            if (target_train_f1 is not None and target_train_f1 - start_train_f1 > 0)
                            else 100,
                            round(
                                float(target_train_loss/current_train_loss)
                                * 100 if current_train_loss else 0
                            )
                            if target_train_loss is not None
                            else 100,
                        ],
                    )
                if percent > last_progress:
                    last_progress = percent
                print_progress(last_progress, f"Training {self.model_name.split('/')[-1]}:")

            i += 1
            self.trained = True
            trained = True
            done=True
            if target_train_f1 is not None:
                if current_train_f1 < target_train_f1:
                    done=False
            if target_train_loss is not None:
                if current_train_loss > target_train_loss:
                    done=False
            if done:
                break
        self.tot_itns = i
        if trained:
            end_progress()
            mins = math.floor((time.time() - st) / 60)
            seconds = round(time.time() - st - mins * 60)
            print(
                f'Training time: {str(mins)}:{str(seconds).rjust(2,"0")}min ({num_train_it} iterations, F1: {round(current_train_f1,3)})'
            )
        torch.cuda.empty_cache()

    def encode_string(self, string, model_name=None) -> torch.Tensor:
        if model_name is None:
            model_name = self.model_name
        use_string = string
        if not isinstance(string, list):
            use_string = [string]
        to_encode = []
        encoding_dict = {}
        to_encode_hash_values = []
        idx_map = {}
        j = 0
        for i, s in enumerate(use_string):
            hash_value = cache_hash(f"{s}_{model_name}")
            value = self.string_encode_cache.get(hash_value, default_value=None)
            encoding_dict[i] = value
            if value is None:
                to_encode.append(s)
                to_encode_hash_values.append(hash_value)
                idx_map[j] = i
                j += 1
        response = None
        if to_encode:
            try:
                self._set_sentence_transformers_model(model_name)
                response = self.model.encode(
                    to_encode,
                    normalize_embeddings=True,
                    show_progress_bar=len(to_encode) > 1000,
                )
            except:
                self._set_transformers_model(model_name)
                response = self._transformers_encode(to_encode, model_name, show_progress_bar=len(to_encode) > 1000)
            if response is not None:
                for i in range(len(to_encode)):
                    self.string_encode_cache.put(to_encode_hash_values[i], response[i])
                    encoding_dict[idx_map[i]] = response[i]
        if not isinstance(string, list):
            return torch.from_numpy(encoding_dict[0])
        embeddings = []
        for i in range(len(use_string)):
            embeddings.append(encoding_dict[i])
        embeddings = np.stack(embeddings)
        return torch.from_numpy(embeddings)
