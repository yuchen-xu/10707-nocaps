from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.readers import CocoCaptionsReader, ConstraintBoxesReader, ImageFeaturesReader
from updown.types import (
    EvaluationInstance,
    EvaluationInstanceWithConstraints,
    EvaluationBatch,
    EvaluationBatchWithConstraints,
)
# from updown.utils.constraints import ConstraintFilter, FiniteStateMachineBuilder
from transformers import BertTokenizer, default_data_collator

class TrainingDataset(Dataset):
    r"""
    A PyTorch `:class:`~torch.utils.data.Dataset` providing access to COCO train2017 captions data
    for training :class:`~updown.models.updown_captioner.UpDownCaptioner`. When wrapped with a
    :class:`~torch.utils.data.DataLoader`, it provides batches of image features and tokenized
    ground truth captions.

    .. note::

        Use :mod:`collate_fn` when wrapping with a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    captions_jsonpath: str
        Path to a JSON file containing COCO train2017 caption annotations.
    image_features_h5path: str
        Path to an H5 file containing pre-extracted features from COCO train2017 images.
    max_caption_length: int, optional (default = 20)
        Maximum length of caption sequences for language modeling. Captions longer than this will
        be truncated to maximum length.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(
        self,
        captions_jsonpath: str,
        image_features_h5path: str,
        max_caption_length: int = 20,
        in_memory: bool = False,
    ) -> None:
        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._captions_reader = CocoCaptionsReader(captions_jsonpath)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", 
            padding="max_length", 
            max_length=512, 
            truncation=True)
        self._max_caption_length = max_caption_length

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        return cls(
            image_features_h5path=_C.DATA.TRAIN_FEATURES,
            captions_jsonpath=_C.DATA.TRAIN_CAPTIONS,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            in_memory=kwargs.pop("in_memory"),
        )

    def __len__(self) -> int:
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)

    def __getitem__(self, index: int):
        if index % 2 == 1:
            image_id, caption = self._captions_reader[index]
            image_features = self._image_features_reader[image_id]
            label = 1
        else:
            _, caption = self._captions_reader[index-1]
            image_id, _ = self._captions_reader[index]
            image_features = self._image_features_reader[image_id]
            label = 0

        visual_token_type_ids = np.ones(image_features.shape[:-1], dtype=np.long)
        visual_attention_mask = np.ones(image_features.shape[:-1], dtype=np.float)

        inputs = self.tokenizer(" ".join(caption), return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        inputs.update(
        {
            "input_ids": inputs['input_ids'].squeeze(0),
            "token_type_ids": inputs['token_type_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "visual_embeds": image_features,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "labels": label,
        })
        return inputs


    def collate_fn(self, batch_list):
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        # image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()
        batch = {}
        keys = batch_list[0].keys()
        visual_embeds = torch.from_numpy(
        _collate_image_features([instance["visual_embeds"] for instance in batch_list])
        )
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        for item in batch_list:
            item.pop('visual_embeds')
            item.pop('visual_token_type_ids')
            item.pop('visual_attention_mask')

        batch = default_data_collator(batch_list)
        batch.update({
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        })

        '''
        print(keys)
        for key in keys:
            print(key)
            print(batch_list[0][key].shape)
            if key != "visual_embeds":
                if key == 'input_ids':
                    tokens = torch.tensor(
                    [instance[key].squeeze(0) for instance in batch_list]
                    ).long()
                else:
                    tokens = torch.tensor(
                    [instance[key] for instance in batch_list]
                    ).long()
                batch[key] = tokens
            else:
                # Pad adaptive image features in the batch.
                image_features = torch.from_numpy(
                _collate_image_features([instance["visual_embeds"] for instance in batch_list])
                )
                batch["visual_embeds"] = image_features
        '''
        return batch


class EvaluationDataset(Dataset):
    r"""
    A PyTorch :class:`~torch.utils.data.Dataset` providing image features for inference. When
    wrapped with a :class:`~torch.utils.data.DataLoader`, it provides batches of image features.

    .. note::

        Use :mod:`collate_fn` when wrapping with a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    image_features_h5path: str
        Path to an H5 file containing pre-extracted features from nocaps val/test images.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(self, image_features_h5path: str, in_memory: bool = True) -> None:
        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._image_ids = sorted(list(self._image_features_reader._map.keys()))

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        return cls(image_features_h5path=_C.DATA.INFER_FEATURES, in_memory=kwargs.pop("in_memory"))

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int):
        image_id = self._image_ids[index]
        image_features = self._image_features_reader[image_id]

        item: EvaluationInstance = {"image_id": image_id, "image_features": image_features}
        return item

    def collate_fn(self, batch_list):
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(
            _collate_image_features([instance["image_features"] for instance in batch_list])
        )

        batch: EvaluationBatch = {"image_id": image_id, "image_features": image_features}
        return batch


class EvaluationDatasetWithConstraints(EvaluationDataset):
    r"""
    A PyTorch :class:`~torch.utils.data.Dataset` providing image features for inference, along
    with constraints for :class:`~updown.modules.cbs.ConstrainedBeamSearch`. When wrapped with a
    :class:`~torch.utils.data.DataLoader`, it provides batches of image features, Finite State
    Machines built (per instance) from constraints, and number of constraints used to make these.

    Extended Summary
    ----------------
    Finite State Machines as represented as adjacency matrices (Tensors) with state transitions
    corresponding to specific constraint (word) occurrence while decoding). We return the number
    of constraints used to make an FSM because it is required while selecting which decoded beams
    satisfied constraints. Refer :func:`~updown.utils.constraints.select_best_beam_with_constraints`
    for more details.

    .. note::

        Use :mod:`collate_fn` when wrapping with a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    image_features_h5path: str
        Path to an H5 file containing pre-extracted features from nocaps val/test images.
    boxes_jsonpath: str
        Path to a JSON file containing bounding box detections in COCO format (nocaps val/test
        usually).
    wordforms_tsvpath: str
        Path to a TSV file containing two fields: first is the name of Open Images object class
        and second field is a comma separated list of words (possibly singular and plural forms
        of the word etc.) which could be CBS constraints.
    hierarchy_jsonpath: str
        Path to a JSON file containing a hierarchy of Open Images object classes as
        `here <https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html>`_.
    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_features_h5path: str,
        boxes_jsonpath: str,
        wordforms_tsvpath: str,
        hierarchy_jsonpath: str,
        nms_threshold: float = 0.85,
        max_given_constraints: int = 3,
        max_words_per_constraint: int = 3,
        in_memory: bool = True,
    ):
        super().__init__(image_features_h5path, in_memory=in_memory)

        self._vocabulary = vocabulary
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")

        self._boxes_reader = ConstraintBoxesReader(boxes_jsonpath)

        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_constraints
        )
        self._fsm_builder = FiniteStateMachineBuilder(vocabulary, wordforms_tsvpath, max_given_constraints, max_words_per_constraint)

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_features_h5path=_C.DATA.INFER_FEATURES,
            boxes_jsonpath=_C.DATA.CBS.INFER_BOXES,
            wordforms_tsvpath=_C.DATA.CBS.WORDFORMS,
            hierarchy_jsonpath=_C.DATA.CBS.CLASS_HIERARCHY,
            max_given_constraints=_C.DATA.CBS.MAX_GIVEN_CONSTRAINTS,
            max_words_per_constraint=_C.DATA.CBS.MAX_WORDS_PER_CONSTRAINT,
            in_memory=kwargs.pop("in_memory"),
        )

    def __getitem__(self, index: int) -> EvaluationInstanceWithConstraints:
        item: EvaluationInstance = super().__getitem__(index)

        # Apply constraint filtering to object class names.
        constraint_boxes = self._boxes_reader[item["image_id"]]

        candidates: List[str] = self._constraint_filter(
            constraint_boxes["boxes"], constraint_boxes["class_names"], constraint_boxes["scores"]
        )
        fsm, nstates = self._fsm_builder.build(candidates)

        return {"fsm": fsm, "num_states": nstates, "num_constraints": len(candidates), **item}

    def collate_fn(
        self, batch_list: List[EvaluationInstanceWithConstraints]
    ) -> EvaluationBatchWithConstraints:

        batch = super().collate_fn(batch_list)

        max_state = max([s["num_states"] for s in batch_list])
        fsm = torch.stack([s["fsm"][:max_state, :max_state, :] for s in batch_list])
        num_candidates = torch.tensor([s["num_constraints"] for s in batch_list]).long()

        batch.update({"fsm": fsm, "num_constraints": num_candidates})
        return batch


def _collate_image_features(image_features_list: List[np.ndarray]) -> np.ndarray:
    num_boxes = [instance.shape[0] for instance in image_features_list]
    image_feature_size = image_features_list[0].shape[-1]

    image_features = np.zeros(
        (len(image_features_list), max(num_boxes), image_feature_size), dtype=np.float32
    )
    for i, (instance, dim) in enumerate(zip(image_features_list, num_boxes)):
        image_features[i, :dim] = instance
    return image_features
