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

    def collate_language_pair(key):
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

        ntokens = src_lengths.sum().item()

        prev_output_tokens = merge(
            key,
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length[key] if pad_to_length is not None else None,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        return {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            },
        }

    src_lang_batch = collate_language_pair("source")
    tgt_lang_batch = collate_language_pair("target")

    batch = {"source_lang_batch": src_lang_batch, "target_lang_batch": tgt_lang_batch}
    return batch


class BilanguagePairDataset(LanguagePairDataset):
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
