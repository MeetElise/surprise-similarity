import numpy as np
import math
import torch
import random
import pandas as pd
from surprise_similarity import StringEncoder
from typing import List



def cosine_similarity(a, b):
    unity = torch.Tensor([1.0]).to(b.device).type(a.type())
    if not torch.isclose(a[0].norm(), unity):
        a = (a.T / a.norm(dim=1)).T
    if not torch.isclose(b[0].norm(), unity):
        b = (b.T / b.norm(dim=1)).T
    return torch.matmul(a, b.T)



class SurpriseSimilarity:
    """
    This class implements the Surprise Similarity Score introduced in ARXIV

    Args: 
        transformer_name: str, default="sentence-transformers/sentence-t5-base"
            Name of a  sentence-transformer model to use for encoding strings

    Main methods:
        - train: trains the sentence transformer model to increase similarity for embeddings specified in the training data
        - predict: returns results for surprise-similarity classification of keys using queries as labels
        - surprise_sim_score: returns surprise similarity score matrix for keys and queries
        - rank documents: efficiently rank a list of documents based on surprise similarity score with a query or list of queries
    """

    def __init__(self, transformer_name="sentence-transformers/sentence-t5-base") -> None:
        self.score_functions = {"cosine": cosine_similarity}
        self.use_keys_cache = None
        self.keys_cache = None
        self.use_keys_keys_map_cache = None
        self.use_keys_embs_cache = None
        self.sample_num_cutoff_cache = None
        self.transformer_name_cache = None
        self.transformer_name = transformer_name
        self.max_itns = False
        self.string_encoder = StringEncoder(model_name=transformer_name)

    def to(self, device):
        self.string_encoder.to(device)
        
    @property
    def device(self):
        return self.string_encoder.device

    def reset_model(self):
        self.use_keys_cache = None
        self.keys_cache = None
        self.use_keys_keys_map_cache = None
        self.use_keys_embs_cache = None
        self.sample_num_cutoff_cache = None
        self.transformer_name_cache = None
        self.string_encoder = StringEncoder(model_name=self.transformer_name)

    def _clear_cache(self):
        self.use_keys_cache = None
        self.keys_cache = None
        self.use_keys_keys_map_cache = None
        self.use_keys_embs_cache = None
        self.sample_num_cutoff_cache = None
        self.transformer_name_cache = None

    def encode_string(self, string, model_name=None) -> torch.Tensor:
        if model_name is None:
            model_name = self.string_encoder.model_name
        return self.string_encoder.encode_string(string=string, model_name=model_name)

    @staticmethod
    def equality(obj1, obj2):
        if obj1 is obj2:
            return True
        if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
            return np.array_equal(obj1, obj2)
        elif type(obj1) != type(obj2):
            return False
        elif isinstance(obj1, torch.Tensor):
            return torch.equal(obj1, obj2)
        return obj1 == obj2

    @staticmethod
    def ensure_tensor_and_device(x, device, dtype=torch.float32):
        if isinstance(x, list):
            x = np.asarray(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(dtype)
        if not x.device == device:
            x = x.to(device)
        return x

    def compute_normalized_scores(
        self,
        key_embs,
        query_embs,
        ensemble_embs=None,
        min_ensemble_size: int=10,
        device: str="cpu",
        dtype: torch.dtype=torch.float32,
        score_function_name: str="cosine",
        return_raw_scores: bool=False,
    ):
        """
        Returns score matrix, scaled such that the mean is 0.5, while maintaining score = 0 -> 0 and score = 1 -> 1
        in the case that the query*ensemble size >1e6, we use an apporximate scaling for efficiency.
        Args:
            key_embs: array-like (List, numpy array, or torch tensor) with dimensions n_key x emb_dim
                the embeddings of the keys
            query_embs:  array-like (List, numpy array, or torch tensor) with dimensions n_query x emb_dim
                the embeddings of the queries
            ensemble_embs: array-like (List, numpy array, or torch tensor) with dimensions n_ensemble x emb_dim
                the embeddings of the ensemble over which to scale such that mean is 0.5, default=None will use keys
            min_ensemble_size: int, default=10
                if ensemble (or keys if ensemble_embs=None) is smaller than this, will return raw scores
            device: str, default="cpu"
                location of embedding tensors
            dtype: torch.dtype, default=torch.float32
                datatype of embedding tensors
            score_function_name: str, default="cosine"
                name of score function to use (currently only cosine is supported)
            return_raw_scores: bool, default=False
                if True, will return raw scores in addition to normalized scores

        Returns: matrix contianing surprise similarity score with dim n_key x n_query
            (and optionally the corresponding raw similarity score matrix if return_raw_scores=True)
        """
        score_function = self.score_functions[score_function_name]
        key_embs = self.ensure_tensor_and_device(key_embs, device, dtype)
        query_embs = self.ensure_tensor_and_device(query_embs, device, dtype)
        raw_scores = score_function(key_embs, query_embs)

        # If keys/ensemble is too small, return raw scores
        if ensemble_embs is not None:
            if len(ensemble_embs) < min_ensemble_size:
                if return_raw_scores:
                    return raw_scores, raw_scores
                return raw_scores
        elif len(key_embs) < min_ensemble_size:
            if return_raw_scores:
                return raw_scores, raw_scores
            return raw_scores

        if ensemble_embs is not None:
            use_scores = score_function(ensemble_embs, query_embs)
        else:
            use_scores = raw_scores

        # To avoid costly calculation of quantiles for large sets, we use an approximation
        if use_scores.numel() > 1e6:
            mean_raw_score = torch.mean(use_scores)
            shifted_raw_scores = raw_scores - mean_raw_score
            heaviside = raw_scores > mean_raw_score
            norm_raw_scores = (
                shifted_raw_scores * (heaviside * (0.5 / (1 - mean_raw_score)) + (~heaviside) * (0.5 / mean_raw_score))
                + 0.5
            )

        else:
            median_raw_score = torch.quantile(use_scores, q=0.5) 
            heaviside = raw_scores > median_raw_score
            mean_raw_score_m = torch.mean(raw_scores[~heaviside])
            mean_raw_score_p = torch.mean(raw_scores[heaviside])
            denominator = mean_raw_score_m - median_raw_score*(mean_raw_score_m + mean_raw_score_p - 1)
            alphap = mean_raw_score_m/denominator
            beta = 1 - alphap
            alpha = (1 - mean_raw_score_p)/denominator
            norm_raw_scores = alpha * raw_scores *(~heaviside) + (alphap * raw_scores + beta) * heaviside
        norm_raw_scores[norm_raw_scores < 0] = 0.0

        if return_raw_scores:
            return norm_raw_scores, raw_scores
        return norm_raw_scores

    def transform_list_of_strings_or_embeddings_to_embeddings(
        self, obj,
        model_name=None,
        dtype=torch.float32,
        device="cpu"
    ) -> torch.Tensor:
        if model_name is None:
            model_name = self.string_encoder.model_name
        if isinstance(obj, str):
            obj = [obj]
        elif isinstance(obj, set):
            obj = list(obj)
        elif isinstance(obj, np.ndarray) and len(obj.shape) == 1:
            len_obj = obj.shape[0]
            obj = obj.reshape(1, len_obj)
        elif isinstance(obj[0], np.ndarray):
            obj = np.asarray(obj)
        if isinstance(obj, list):
            obj = self.string_encoder.encode_string(obj, model_name=model_name)
        obj = self.ensure_tensor_and_device(obj, device=device, dtype=dtype)
        return obj

    def compute_ensemble_query_median_1sigma(
        self,
        ensemble_embs,
        query_embs,
        device="cpu",
        use_percentile=False,
        score_function_name="cosine",
        dtype=torch.float32,
        return_scores=False,
    ):
        normalized_scores, raw_scores = self.compute_normalized_scores(
            ensemble_embs,
            query_embs,
            device=device,
            score_function_name=score_function_name,
            dtype=dtype,
            return_raw_scores=True,
        )
        if use_percentile:
            cpu_scores = raw_scores.T.cpu().type(dtype).numpy()
            all_means = np.percentile(cpu_scores, 50, axis=1)
            all_stds = torch.from_numpy(
                (np.percentile(cpu_scores, 100 * (1 + math.erf(1 / math.sqrt(2))) / 2, axis=1) - all_means)
            )
            all_means = torch.from_numpy(all_means)
        else:
            all_means = torch.mean(raw_scores.T, axis=1)
            all_stds = torch.std(raw_scores.T, axis=1, unbiased=False)
        avgs_stds = torch.concat([all_means.view(-1, 1), all_stds.view(-1, 1)], dim=1)
        avgs_stds = self.ensure_tensor_and_device(avgs_stds, device=device, dtype=dtype)
        if return_scores:
            return avgs_stds, normalized_scores, raw_scores
        return avgs_stds

    def train(
        self,
        keys: List[str],
        queries: List[str],
        shuffle: bool=True,
        target_train_f1: float=0.9,
        target_train_loss: float=0.3,
        max_its: int=70,
        min_its: int=0,
        template: str='this matter is {}',
        show_progress: bool=False,
        labels: List[str]=None,
        lr_factor: float=5,
        loss: str="xCos",
        gamma: float=1,
        weight_decay: float=0.01,
    ):
        """
        Train will fine-tune the sentence-transformer model with new data. 
        
        Args:
            keys: List[str]
                Objects to classify: make up the input of the training data
            queries: List[str]
                The target labels with which to classify the keys
                None in the queries list indicates that a sample is not classified as any label
            shuffle: bool, default=True
                If True, training data will be shuffled
            target_train_f1: float, default=0.9
                If the model achieves this F1 score on the training data, training will stop
                If None, training will continue until max_its or target_train_loss is reached
            target_train_loss: float, default=0.3
                If the model achieves this loss on the training data, training will stop
                If None, training will continue until max_its or target_train_f1 is reached
            max_its: int, default=70
                The maximum number of iterations to train
            min_its: int, default=0
                The minimum number of iterations to train
            template: str, default="this matter is {}"
                A mapping from input queries to queries to be embedded, must contain a single '{}' which will be 
                filled by an item from queries.  In None, queries will be embedded directly.
            show_progress: bool, default=False
                If True, will show a progress updates during training
            labels: List[str], default=None
                The list of possible labels to be assigned. If None, will be inferred from queries
            lr_factor: float, default=5
                The learning rate will be set to lr_factor * 2e-5
            loss: str, default="xCos"
                The loss to use during training.  Currently supported losses are: "xCos" and "CosS"
                xCos: the focal cross entropy loss (https://arxiv.org/abs/1708.02002v2)
                CosS: the cosine similarity loss (https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss)
            gamma: float, default=1
                gamma parameter for the focal loss
            weight_decay: float, default=0.01   
        """
        if template is not None:
            queries = [
                template.format(qry) if not isinstance(qry, list) else [template.format(itm) for itm in qry]
                for qry in queries
            ]
        assert len(keys) == len(queries), "keys and queries must be same length"
        self.string_encoder.train(
            keys,
            queries,
            shuffle=shuffle,
            target_train_f1=target_train_f1,
            target_train_loss=target_train_loss,
            max_its=max_its,
            show_progress=show_progress,
            labels=labels,
            min_its=min_its,
            lr_factor=lr_factor,
            loss=loss,
            gamma=gamma,
            weight_decay=weight_decay,
        )
        self._clear_cache()
        self.max_itns = self.string_encoder.max_itns
        self.tot_itns = self.string_encoder.tot_itns

    def predict(
        self,
        keys: List[str],
        queries: List[str],
        ensemble: List[str]=None,
        template: str="this matter is {}",
        surprise_weight: float=None,
        Ncross: int=1000,
        non_exclusive_threshold: float=0.5,
        exclusive: bool=True,
        return_scores: bool=False
    ):
        """
        Predict uses queries as a label-set to be assigned to keys using the similarity score.

        Args:
            keys: List[str]
                Objects to classify
            queries: List[str]
                The labels with which to classify the keys
            ensemble: List[str]
                The ensemble which provides context for query raw similarity scores, default=None will use keys
            template: str, default="this matter is {}"
                A mapping from input queries to queries to be embedded, must contain a single '{}' which will be 
                filled by an item from queries.  In None, queries will be embedded directly.
            surprise_weight: float, default=None
                The weight to give to the surprise score relative to the raw score, should be between 0 and 1 (inclusive.)
                If None, will be set to tanh(len(ensemble)/Ncross)
            Ncross: int, default=1000
                The ensemble size which will correspond to surprise weight=tanh(1).
            non_exclusive_threshold: float, default=0.5
                The surprise score threshold above which a key will be labeled as a query for non-exclusive classification.
            exclusive: bool, default=True
                Whether to use exclusive or non-exclusive classification.
            return_scores: bool, default=False
                If True, surprise scores will be returned in addition to the predictions.
        """

        if ensemble is None:
            ensemble = keys
        if surprise_weight is None:
            surprise_weight = math.tanh(len(ensemble) / Ncross)
            if surprise_weight > 0.99:
                surprise_weight = 1
            elif surprise_weight < 0.01:
                surprise_weight = 0
        len_input_queries = len(queries)
        if StringEncoder.check_binary_negative(queries):
            exclusive = False
        labels = queries
        if template is not None:
            queries = [template.format(qry) for qry in queries]
        label_dict = {i: labels[i] for i in range(len(labels)) if labels[i] is not None}
        surprise_scores = self.ensure_tensor_and_device(
            self.surprise_sim_score(keys=keys, queries=queries, ensemble=ensemble, surprise_weight=surprise_weight),
            "cpu",
        ).numpy()
        if not exclusive and len_input_queries == 1:
            surprise_preds_results = []
            for score in surprise_scores[:, 0]:
                if score >= non_exclusive_threshold:
                    surprise_preds_results.append(label_dict[0])
                else:
                    surprise_preds_results.append(None)
        elif not exclusive:
            surprise_preds_results = []
            for score in surprise_scores:
                tmp = []
                for idx in range(len(score)):
                    if score[idx] >= non_exclusive_threshold:
                        tmp.append(label_dict[idx])
                surprise_preds_results.append(tmp)
        else:
            surprise_preds_results = [label_dict[itm] for itm in np.argmax(surprise_scores, axis=1)]
        if return_scores:
            return surprise_preds_results, surprise_scores
        return surprise_preds_results

    @staticmethod
    def get_max_batch_size(dtype=torch.float32):
        """
        Rough estimate for the max batch size that the surprise_sim_score can handle. Excludes model size.
        """
        if not torch.cuda.device_count():
            return 512
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        if dtype == torch.float16:
            return round(math.sqrt(0.9 * (t - r) / 10**6 / (7721 - 1660)) * 20000)
        if dtype == torch.float32:
            return round(math.sqrt(0.9 * (t - r) / 10**6 / (11541 - 1660)) * 20000)

    @staticmethod
    def preserve_surprise_ordering_at_high_sigma(raw_surprise_scores,
                         threshold_sigma=4.5,
                         new_max_sigma=5.3,
                         device=None):
        """
        This function modifies the tail of the distribution of cosine scores in such a way that 
        that the resulting surprise score can still discriminate between objects in the tail. (Otherwise
        for any value more than 5.4 sigma away the surprise score gives back 1.0). The modification is a
        simple overall linear transformation which only affects the cosine scores which are more the
        threshold sigma away from the mean. 

        Args:
            raw_surprise_scores: torch.Tensor
                The z-scores of the cosine score distributions (the number ofsigma away from the mean of each data point)
            threshold_sigma: float, default=4.5
                Defines the threshold above which the linear rescaling is applied.
            new_max_sigma: float, default=5.3
                Defines the highest number or sigmas away from the mean in the shifted distribution.  

        Returns: torch.Tensor
        """
        threshold_sigma = threshold_sigma*(0.5**0.5)
        new_max_sigma = new_max_sigma*(0.5**0.5)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        max_sigma = torch.max(raw_surprise_scores).to(device=device)
        shifted_scores=torch.add(
            torch.mul(
                raw_surprise_scores, 
                (new_max_sigma - threshold_sigma)/(max_sigma - threshold_sigma)),
                threshold_sigma*(max_sigma - new_max_sigma)/(max_sigma - threshold_sigma)
                )
        return torch.where(raw_surprise_scores > threshold_sigma, shifted_scores, raw_surprise_scores)
    
    def surprise_sim_score(
        self,
        keys: List[str],
        queries: List[str],
        ensemble: List[str]=None,
        surprise_weight: int=None,
        Ncross: int=1000,
        use_percentile: bool=False,
        transformer_name: str=None,
        ensemble_query_median_1sigma: torch.Tensor=None,
        return_ensemble_query_median_1sigma: bool=False,
        preserve_surprise_ordering_at_high_sigma: bool=True,
        score_function_name: str="cosine",
        normalize_raw_similarity: bool=True,
        device: str=None,
        dtype: torch.dtype=torch.float32,
    ):
        """
        Compute the surprise score matrix. Dimensions = n_key x n_queries.

        Args:
            keys: List[str]
            queries: List[str]
            ensemble: List[str]
                The ensemble which provides context for query raw similarity scores, default=None will use keys
            surprise_weight: float, default=None
                The weight to give to the surprise score relative to the raw score, should be between 0 and 1 (inclusive.)
                If None, will be set to tanh(len(ensemble)/Ncross)
            Ncross: int, default=1000
                The ensemble size which will correspond to surprise weight=tanh(1).
            use_percentile: bool, default=False
                If True, will use the 50th and 84th percentile of the raw similarity scores as the mean and 1 sigma
            transformer_name: str, default=None
                The name of the transformer to use. If None, will use the class transformer.
            ensemble_query_median_1sigma: torch.Tensor, default=None
                Saved values of the mean and 1 sigma of the raw similarity scores for queries over the ensemble.
                If None, will be computed.
            return_ensemble_query_median_1sigma: bool, default=False
                If True, will return the ensemble_query_median_1sigma in addition to the surprise score matrix.
            preserve_surprise_ordering_at_high_sigma: bool, default=True
                If True, will modify the surprise scores in the tail of the distribution so that they can still
                be distinguished, preserving their order.  Otherwise normal precision results in all tail values
                being mapped to 1.0.
            score_function_name: str, default="cosine"
                Only cosine is supported at the moment.
            normalize_raw_similarity: bool, default=True
                If True, will normalize the raw similarity scores to have mean = 0.5 even in the case surprise_weight=0.
            device: str, default=None
                The device to use. If None, will use cuda:0 if available, otherwise cpu.
            dtype: torch.dtype, default=torch.float32
                The data type to use.

        """
        if ensemble is None:
            ensemble = keys
        if surprise_weight is None:
            surprise_weight = math.tanh(len(ensemble) / Ncross)
            if surprise_weight < 0.01:
                surprise_weight = 0
            elif surprise_weight > 0.99:
                surprise_weight = 1
                
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # flattened list of objects in keys, queries and ensemble
        original = []
        # flattened list of embeddings of objects in keys, queries and ensemble
        transformed = []
        for obj in [keys, queries, ensemble]:
            if obj is None or obj == []:
                transformed.append(obj)
                original.append(obj)
                continue
            found = False
            for i, it in enumerate(original):
                if self.equality(obj, it):
                    transformed.append(transformed[i])
                    original.append(obj)
                    found = True
                    break
            if not found:
                transformed.append(
                    self.transform_list_of_strings_or_embeddings_to_embeddings(
                        obj, model_name=transformer_name, dtype=dtype, device=device
                    )
                )
                original.append(obj)

        keys, queries, ensemble = transformed
        normalized_key_query_scores = None
        # if the surprise weight is 0, we don't need ensemble statistics
        if surprise_weight == 0 or len(ensemble) == 0:
            if normalize_raw_similarity:
                return self.compute_normalized_scores(
                    keys,
                    queries,
                    ensemble_embs=ensemble,
                    device=device,
                    dtype=dtype,
                    score_function_name=score_function_name
                    )
            else:
                return self.compute_normalized_scores(
                    keys,
                    queries,
                    ensemble_embs=ensemble,
                    device=device,
                    dtype=dtype,
                    score_function_name=score_function_name,
                    return_raw_scores=True
                )[1]
        # Calculate ensemble statistics if they are not provided
        if ensemble_query_median_1sigma is None:
            if self.equality(keys, ensemble):
                (
                    ensemble_query_median_1sigma,
                    normalized_key_query_scores,
                    key_query_scores,
                ) = self.compute_ensemble_query_median_1sigma(
                    ensemble,
                    queries,
                    device=device,
                    use_percentile=use_percentile,
                    dtype=dtype,
                    return_scores=True,
                    score_function_name=score_function_name,
                )
            else:
                ensemble_query_median_1sigma = self.compute_ensemble_query_median_1sigma(
                    ensemble,
                    queries,
                    device=device,
                    use_percentile=use_percentile,
                    dtype=dtype,
                    score_function_name=score_function_name,
                )
        else:
            assert ensemble_query_median_1sigma.shape == (2, len(queries))
        # Calculate key-query scores if they are not included in the ensemble-query scores
        if normalized_key_query_scores is None:
            normalized_key_query_scores, key_query_scores = self.compute_normalized_scores(
                keys,
                queries,
                ensemble_embs=ensemble,
                device=device,
                dtype=dtype,
                score_function_name=score_function_name,
                return_raw_scores=True
            )
        # the raw surprise is the z-score: number of std-dev of each score from the mean
        raw_surprise_scores = (key_query_scores - ensemble_query_median_1sigma[:, 0]) / (
            ensemble_query_median_1sigma[:, 1]
        )
        # shift the tails of the distribution to a new high sigma to preserve ordering in the tails
        if torch.max(raw_surprise_scores)>5.4 and preserve_surprise_ordering_at_high_sigma:
            raw_surprise_scores = self.preserve_surprise_ordering_at_high_sigma(
                raw_surprise_scores,
                device=device
                )
        if surprise_weight == 1:
            return self.sim_exponent_to_sim_score(raw_surprise_scores)
        surp_sim_score = (
            1 - surprise_weight
        ) * normalized_key_query_scores + surprise_weight * self.sim_exponent_to_sim_score(raw_surprise_scores)
        if return_ensemble_query_median_1sigma:
            return surp_sim_score, ensemble_query_median_1sigma
        return surp_sim_score

    @staticmethod
    def sim_exponent_to_sim_score(sim_exponent):
        return (1 + torch.erf(sim_exponent / 2**0.5)) / 2

    def rank_documents(
        self,
        documents: list,
        queries: list,
        surprise_weight: float=None,
        Ncross: int=1000,
        batch_size=20000,
        upper_thresholds: dict = None,
        lower_thresholds: dict = None,
        cutoffs: list = None,
        sample_num_cutoff: int = 100,
        search_factor=10,
        sort_by_query: str = None,
        show_progress_bar=True,
        normalize_raw_similarity=True,        
        transformer_name=None,
        device=None,
        dtype=torch.float32,
    ):
        """
        Return a sorted pandas dataframe containing documents (keys) and their associated surprise score for each query.

        Args:
            documents: List[str]
                List of documents to rank
            queries: List[str]
                list of queries: the documents by similarity to each query individually
            surprise_weight: float, default=None
                The weight to give to the surprise score relative to the raw score, should be between 0 and 1 (inclusive.)
                If None, will be set to tanh(len(batch_size)/Ncross)
            Ncross: int, default=1000
                The ensemble size which will correspond to surprise weight=tanh(1).
            batch_size: int, default=20000
                The document batch size to use when computing the surprise score.
            upper(lower)_thresholds: dict, default = None
                A dictionary used to filter the results s.t. documents with similarity scores greater(less) than 
                the specified value for each query are excluded.  Dictionary keys are a subset of the queries
            cutoffs: List[str], default = None
                Results will be returned only for documents more similar to each query than the cuttoff is similar
                to the query.  If provided, this list must contain one cutoff per query.
            sample_num_cutoff: int, default = 100
                The returned dataframe will contain the top sample_num_cutoff documents for each query.
                If None, all documents will be returned.
            search_factor: int, default = 10
                If sample_num_cutoff is not None, the surprise score will be evaluated over the documents with the
                top (sample_num_cutoff * search_factor) raw scores.
            sort_by_query: str, default = None
                The results dataframe will be sorted by similarity to this query.  If None, queries[0] will be used.
            show_progress_bar: bool, default = True
                If True, a progress bar will be displayed.
            normalize_raw_similarity: bool, default = True
                If True, the raw similarity scores will be linearly scaled to have mean = 0.5 while mapping 0->0 and 1->1.
                Otherwise, raw scores will be used.
            transformer_name: str, default=None
                The name of the transformer to use. If None, will use the class transformer.     
            device: str, default=None
                The device to use. If None, will use cuda:0 if available, otherwise cpu.
            dtype: torch.dtype, default=torch.float32
                The data type to use.                           
            


        """
        keys = documents
        if transformer_name is None:
            transformer_name = self.string_encoder.model_name
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(cutoffs, str):
            cutoffs = [cutoffs]
        if upper_thresholds is not None:
            assert set(upper_thresholds.keys()).issubset(
                set(queries)
            ), "All keys of upper_thresholds must be contained in queries"
        if lower_thresholds is not None:
            assert set(lower_thresholds.keys()).issubset(
                set(queries)
            ), "All keys of lower_thresholds must be contained in queries"

        max_batch_size = self.get_max_batch_size(dtype)
        if batch_size > self.get_max_batch_size(dtype):
            print(f"Batch size {batch_size} too large for GPU, reducing to max batch size {max_batch_size}")
            batch_size = max_batch_size
        if sort_by_query is None:
            sort_by_query = queries[0]
        if cutoffs is None:
            cutoffs = []
        if cutoffs:
            assert len(queries) == len(cutoffs), "If cutoffs are provided, they must be the same length as queries"
            cutoff_query_dict = {cutoff: query for cutoff, query in zip(cutoffs, queries)}
            query_cutoff_dict = {query: cutoff for cutoff, query in zip(cutoffs, queries)}
        queries_and_cutoffs = list(set(queries + cutoffs))
        assert (
            2 * len(queries_and_cutoffs) < batch_size
        ), "Batch size too small to accommodate {len(queries_and_cutoffs)} queries and cutoffs"
        queries_and_cutoffs_embeddings = self.string_encoder.encode_string(
            queries_and_cutoffs, model_name=transformer_name
        )

        if self.equality(keys, self.keys_cache) and self.transformer_name_cache == transformer_name:
            use_keys_embeddings = self.use_keys_embs_cache
            use_keys = self.use_keys_cache
            use_keys_keys_map = self.use_keys_keys_map_cache
        else:
            use_keys_keys_map = {}
            use_keys = []
            j = 0
            for idx in random.sample(range(0, len(keys)), len(keys)):
                key = keys[idx]
                if use_keys_keys_map.get(key) is None:
                    use_keys.append(key)
                    use_keys_keys_map[j] = idx
                    j += 1
            use_keys_embeddings = self.string_encoder.encode_string(use_keys, model_name=transformer_name)
            self.use_keys_embs_cache = use_keys_embeddings
            self.use_keys_cache = use_keys
            self.use_keys_keys_map_cache = use_keys_keys_map
            self.keys_cache = keys
            self.transformer_name_cache = transformer_name

        masked_use_keys = use_keys
        masked_use_keys_embs = use_keys_embeddings
        masked_use_keys_keys_map = use_keys_keys_map
        if sample_num_cutoff:
            if len(use_keys_embeddings) > sample_num_cutoff * search_factor:
                target_scores = cosine_similarity(
                    use_keys_embeddings,
                    queries_and_cutoffs_embeddings[[i for i, val in enumerate(queries_and_cutoffs) if val in queries]],
                )

                # sample_num_cutoff_score is a value indicating the separation between the kth value and the rest.
                sample_num_cutoff_score = -torch.kthvalue(
                    -target_scores, k=min(len(target_scores), max(1000, sample_num_cutoff * search_factor)), dim=0
                ).values

                # This counts the number of keys per query which have a score larger than the cutoff
                sample_num_mask = (target_scores >= sample_num_cutoff_score).sum(axis=1) > 0
                masked_use_keys_embs = use_keys_embeddings[sample_num_mask]
                masked_use_keys = np.asanyarray(use_keys)[sample_num_mask].tolist()
                masked_use_keys_keys_map = {
                    i: use_keys_keys_map[val] for i, val in enumerate(torch.nonzero(sample_num_mask).squeeze().tolist())
                }

        n_batches = math.ceil(len(masked_use_keys_embs) / (batch_size - len(queries_and_cutoffs_embeddings)))

        # Reset batch size so that the batches have roughly equal size for stats reasons
        batch_size = len(masked_use_keys_embs) / n_batches
        batch_size = math.ceil(batch_size)
        if surprise_weight is None:
            surprise_weight = math.tanh(batch_size / Ncross)
            if surprise_weight < 0.01:
                surprise_weight = 0
        from tqdm import trange
        cutoff_scores_dict = {}
        scores_dict = {}
        if n_batches > 20 and show_progress_bar:
            iterator = trange(n_batches, desc="Ranking")
        else:
            iterator = range(n_batches)
        for i in iterator:
            if len(masked_use_keys_embs[i * batch_size : (i + 1) * batch_size]) == 0:
                continue
            batch_all_embeddings = torch.cat(
                [queries_and_cutoffs_embeddings, masked_use_keys_embs[i * batch_size : (i + 1) * batch_size]], dim=0
            )
            query_surprise_scores = self.surprise_sim_score(
                keys=batch_all_embeddings,
                queries=batch_all_embeddings,
                surprise_weight=surprise_weight,
                normalize_raw_similarity=normalize_raw_similarity,
                device=device,
                dtype=dtype,
            )
            if cutoffs:
                for cutoff in cutoffs:
                    if not cutoff in cutoff_scores_dict:
                        cutoff_scores_dict[cutoff] = []
                    tmp_cutoff = query_surprise_scores[
                        queries_and_cutoffs.index(cutoff_query_dict[cutoff]), queries_and_cutoffs.index(cutoff)
                    ]
                    cutoff_scores_dict[cutoff].append(float(tmp_cutoff.cpu().type(torch.float32).numpy()))
            for query in queries:
                if not query in scores_dict:
                    scores_dict[query] = []
                scores_dict[query].append(
                    query_surprise_scores[queries_and_cutoffs.index(query), len(queries_and_cutoffs) :]
                    .to("cpu")
                    .type(torch.float32)
                    .numpy()
                )
        for query in queries:
            scores_dict[query] = np.concatenate(scores_dict[query])
        for cutoff in cutoffs:
            cutoff_scores_dict[cutoff] = sum(cutoff_scores_dict[cutoff]) / len(cutoff_scores_dict[cutoff])
        df = pd.DataFrame(masked_use_keys, columns=["documents"])
        scores = np.asanyarray([scores_dict[query] for query in queries]).T
        df[queries] = scores

        if upper_thresholds or lower_thresholds:
            upper_mask = pd.DataFrame(
                {col: df[col] < threshold for col, threshold in upper_thresholds.items()} if upper_thresholds else True,
                index=df.index,
                columns=df.columns,
            ).fillna(True)
            lower_mask = pd.DataFrame(
                {col: df[col] > threshold for col, threshold in lower_thresholds.items()} if lower_thresholds else True,
                index=df.index,
                columns=df.columns,
            ).fillna(True)
            mask = upper_mask & lower_mask
            df = df[mask.all(axis=1)]

        if sample_num_cutoff:
            if len(df) > sample_num_cutoff:
                sample_num_cutoff_score = -torch.kthvalue(-torch.from_numpy(scores), k=sample_num_cutoff, dim=0).values
                sample_num_mask = (scores >= sample_num_cutoff_score.numpy()).sum(axis=1) > 0
                df = df[sample_num_mask]
        if cutoff_scores_dict:
            cutoff_scores = np.asanyarray([cutoff_scores_dict[query_cutoff_dict[query]] for query in queries])
            mask = (df[queries].values > cutoff_scores).sum(axis=1) > 0
            df = df[mask]
        df = df.sort_values(by=sort_by_query, ascending=False)
        df.index = [masked_use_keys_keys_map[val] for val in df.index]
        return df