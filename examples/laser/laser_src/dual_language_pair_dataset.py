from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch

from fairseq.data import data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset

logger = logging.getLogger(__name__)


def dual_language_collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def collate_single_language(key):
        id = torch.LongTensor([s["id"] for s in samples])
        src_tokens = merge(
            key,
            left_pad=left_pad_source,
            pad_to_length=pad_to_length[key] if pad_to_length is not None else None,
        )
        # sort by descending source length
        src_lengths = torch.LongTensor([s[key].ne(pad_idx).long().sum() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        target = merge(
            key,
            left_pad=left_pad_target,
            pad_to_length=pad_to_length[key] if pad_to_length is not None else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s[key].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()
        prev_output_tokens = merge(
            key,
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length[key] if pad_to_length is not None else None,
        )
        prev_output_tokens.index_select(0, sort_order)

        return {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "target": target,
        }

    batch_lang1 = collate_single_language("source")
    batch_lang2 = collate_single_language("target")

    batch = {"batch_lang1": batch_lang1, "batch_lang2": batch_lang2}

    return batch


class DualLanguagePairDataset(LanguagePairDataset):
    def collater(self, samples, pad_to_length=None):
        res = dual_language_collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        return res
