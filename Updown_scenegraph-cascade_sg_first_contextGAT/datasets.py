import torch
from torch.utils.data import Dataset
import h5py
import pickle
import json
import os
from readers import CocoCaptionsReader, ImageFeaturesReader
from typing import Dict, List
import numpy as np

max_img_features = 36

# Todo: data directory might need to be changed

class TrainingDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, vocabulary, captions_jsonpath, image_features_h5path,
                 max_caption_length=20, in_memory=False):
        self._vocabulary = vocabulary
        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._captions_reader = CocoCaptionsReader(captions_jsonpath)
        self._max_caption_length = max_caption_length

        # this part handles the scene graph features
        # notice that scene graph is based on different object detection network
        self.sg_train_h5 = h5py.File(data_folder + '/train_scene-graph.hdf5', 'r')
        self.train_obj = self.sg_train_h5['object_features']
        self.train_obj_mask = self.sg_train_h5['object_mask']
        self.train_rel = self.sg_train_h5['relation_features']
        self.train_rel_mask = self.sg_train_h5['relation_mask']
        self.train_pair_idx = self.sg_train_h5['relation_pair_idx']
        with open(os.path.join(data_folder, 'train_scene-graph_imgid2idx.pkl'), 'rb') as j:
            self.sg_id = pickle.load(j)

    def __getitem__(self, index):
        # so the reader is the key to how we manage feature caption correspondence
        # looks like we also need to access graph features via image id
        image_id, caption = self._captions_reader[index]
        image_features = self._image_features_reader[image_id]
        sg_index = self.sg_id[image_id]

        obj = torch.tensor(self.train_obj[sg_index], dtype=torch.float)
        rel = torch.tensor(self.train_rel[sg_index], dtype=torch.float)
        obj_mask = torch.tensor(self.train_obj_mask[sg_index], dtype=torch.bool)
        rel_mask = torch.tensor(self.train_rel_mask[sg_index], dtype=torch.bool)
        pair_idx = self.train_pair_idx[sg_index]

        # Tokenize caption.
        caption_tokens: List[int] = [self._vocabulary.get_token_index(c) for c in caption]

        # Pad upto max_caption_length.
        caption_tokens = caption_tokens[: self._max_caption_length]
        caplen = len(caption_tokens)
        caplen = torch.tensor([caplen], dtype=torch.long)
        caption_tokens.extend(
            [self._vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self._max_caption_length - len(caption_tokens))
        )

        caption_tokens = torch.tensor(caption_tokens).long()

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(_collate_image_features(image_features))

        item = [image_features, obj, rel, obj_mask, rel_mask, pair_idx, caption_tokens, caplen]
        return item

    def __len__(self) -> int:
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)




class ValidationDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, vocabulary, captions_jsonpath, image_features_h5path,
                 max_caption_length=20, in_memory=False):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """

        self._vocabulary = vocabulary
        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._captions_reader = CocoCaptionsReader(captions_jsonpath)
        self._max_caption_length = max_caption_length

        # this is for validation
        self.sg_val_h5 = h5py.File(data_folder + '/val_scene-graph.hdf5', 'r')
        self.val_obj = self.sg_val_h5['object_features']
        self.val_obj_mask = self.sg_val_h5['object_mask']
        self.val_rel = self.sg_val_h5['relation_features']
        self.val_rel_mask = self.sg_val_h5['relation_mask']
        self.val_pair_idx = self.sg_val_h5['relation_pair_idx']
        with open(os.path.join(data_folder, 'val_scene-graph_imgid2idx.pkl'), 'rb') as j:
            self.sg_id = pickle.load(j)

    def __getitem__(self, index):
        # so the reader is the key to how we manage feature caption correspondence
        # looks like we also need to access graph features via image id
        image_id, caption = self._captions_reader[index]
        image_features = self._image_features_reader[image_id]
        sg_index = self.sg_id[image_id]

        caption_idx = self._captions_reader.img_2_cap[image_id]
        all_caps = []
        for idx in caption_idx:
            all_caps.append(self._captions_reader[idx][1])

        obj = torch.tensor(self.val_obj[sg_index], dtype=torch.float)
        rel = torch.tensor(self.val_rel[sg_index], dtype=torch.float)
        obj_mask = torch.tensor(self.val_obj_mask[sg_index], dtype=torch.bool)
        rel_mask = torch.tensor(self.val_rel_mask[sg_index], dtype=torch.bool)
        pair_idx = self.val_pair_idx[sg_index]

        # Tokenize caption.
        caption_tokens: List[int] = [self._vocabulary.get_token_index(c) for c in caption]

        # Pad upto max_caption_length.
        caption_tokens = caption_tokens[: self._max_caption_length]
        caplen = len(caption_tokens)
        caplen = torch.tensor([caplen], dtype=torch.long)
        caption_tokens.extend(
            [self._vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self._max_caption_length - len(caption_tokens))
        )

        caption_tokens = torch.tensor(caption_tokens).long()

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(_collate_image_features(image_features))

        item = [image_features, obj, rel, obj_mask, rel_mask, pair_idx, caption_tokens, caplen, all_caps]
        return item

    def __len__(self) -> int:
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)




class TestDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, image_features_h5path, in_memory=False):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """

        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)

        # this is for validation
        self.sg_val_h5 = h5py.File(data_folder + '/val_scene-graph.hdf5', 'r')
        self.val_obj = self.sg_val_h5['object_features']
        self.val_obj_mask = self.sg_val_h5['object_mask']
        self.val_rel = self.sg_val_h5['relation_features']
        self.val_rel_mask = self.sg_val_h5['relation_mask']
        self.val_pair_idx = self.sg_val_h5['relation_pair_idx']
        with open(os.path.join(data_folder, 'val_scene-graph_imgid2idx.pkl'), 'rb') as j:
            self.sg_id = pickle.load(j)

    def __getitem__(self, index):
        # so the reader is the key to how we manage feature caption correspondence
        # looks like we also need to access graph features via image id
        image_id = self._image_features_reader.image_id[index]
        image_features = self._image_features_reader[image_id]
        sg_index = self.sg_id[image_id]

        obj = torch.tensor(self.val_obj[sg_index], dtype=torch.float)
        rel = torch.tensor(self.val_rel[sg_index], dtype=torch.float)
        obj_mask = torch.tensor(self.val_obj_mask[sg_index], dtype=torch.bool)
        rel_mask = torch.tensor(self.val_rel_mask[sg_index], dtype=torch.bool)
        pair_idx = self.val_pair_idx[sg_index]

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(_collate_image_features(image_features))

        item = [image_features, obj, rel, obj_mask, rel_mask, pair_idx, image_id]
        return item

    def __len__(self) -> int:
        # Number of training examples are number of captions, not number of images.
        return len(self._image_features_reader.image_id)


def _collate_image_features(image):
    num_box = image.shape[0]
    image_feature_size = image.shape[-1]
    image_features = np.zeros(
        (max_img_features, image_feature_size), dtype=np.float32
    )
    image_features[:num_box] = image[:max_img_features]
    return image_features