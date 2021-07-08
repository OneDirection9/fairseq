from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as osp
import tempfile

from laser_src.data.text_processing import BPEfastApply, Token

# get environment
assert os.environ.get("LASER"), "Please set the environment variable LASER"
LASER = os.environ["LASER"]

ALIAS_TO_CHECKPOINT_NAME = {
    "21": "bilstm.eparl21.2018-11-19.pt",
    "93": "bilstm.93langs.2018-12-26.pt",
}

ALIAS_TO_BPE_CODES_NAME = {
    "21": "eparl21.fcodes",
    "93": "93langs.fcodes",
}

ALIAS_TO_VOCAB_NAME = {
    "21": "eparl21.fvocab",
    "93": "93langs.fvocab",
}

MODEL_DIR = osp.join(LASER, "models")

# a special character indicting the start of the persona
# see tools/convai2.py
PERSONA_SEP_CHAR = "😈"
# TODO: handle PERSONA_SEP_CHAR in dictionary
# To avoid encode PERSONA_SEP_CHAR to UNK word, I add it to
# the last line of 93langs.fvocab temporarily and manually.
# So the PERSONA_SEP_CHAR will be encoded to `73640`.
# After that remove PERSONA_SEP_CHAR from fvocab, to avoid
# size mismatch between model and checkpoint.
PERSONA_SEP_CHAR_ID = 73640


def get_checkpoint(alias: str) -> str:
    return osp.join(MODEL_DIR, ALIAS_TO_CHECKPOINT_NAME[alias])


def get_bpe_codes(alias: str) -> str:
    return osp.join(MODEL_DIR, ALIAS_TO_BPE_CODES_NAME[alias])


def get_vocab(alias: str) -> str:
    return osp.join(MODEL_DIR, ALIAS_TO_VOCAB_NAME[alias])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alias",
        type=str,
        choices=("21", "93"),
        default="93",
        help="number of languages on which model pretrained",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")

    args = parser.parse_args()
    return args


def process(inp_file, out_file, lang, bpe_codes, verbose=False):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = osp.join(tmp_dir, "tok")
        romanize = True if lang == "el" else False
        Token(
            inp_file,
            tmp_file,
            lang=lang,
            romanize=romanize,
            lower_case=True,
            gzip=False,
            verbose=verbose,
            over_write=False,
        )
        BPEfastApply(tmp_file, out_file, bpe_codes, verbose=verbose, over_write=False)


LANG_MAP = {
    "cmn": "zh",
    "eng": "en",
    "ita": "it",
}


def main():
    args = parse_args()

    data_root = osp.join(LASER, "laser_src/data")
    bpe_codes = get_bpe_codes(args.alias)

    def _get_inp_dir(_folder):
        if _folder == "tatoeba":
            return osp.join(data_root, _folder, "v1")
        else:
            return osp.join(data_root, _folder, "raw")

    def _get_oup_dir(_folder):
        return osp.join(data_root, _folder, f"bpe{args.alias}")

    def _get_inp_tmpl(_folder):
        tmpl = str(_folder) + ".{}.{}"
        if _folder == "UNPC":
            tmpl = tmpl + ".2000000"
        return tmpl

    def _get_oup_tmpl(_folder):
        return "train.{}.{}"

    def _get_lang(lang):
        if lang in LANG_MAP:
            return LANG_MAP[lang]
        elif len(lang) == 3 and lang[-1] in {"1", "2"}:  # decode of _to_lang_pair in nli.py
            return lang[:-1]
        else:
            return lang

    datasets = {
        "Europarl": {
            "lang_pairs": ("en-it",),
        },
        "news-commentary": {
            "lang_pairs": ("en-zh",),
        },
        "WikiMatrix": {
            "lang_pairs": ("en-zh",),
        },
        "mt": {
            "lang_pairs": ("en-zh",),
        },
        "UNPC": {
            "lang_pairs": ("en-zh",),
        },
        "news-crawl": {
            "lang_pairs": ("zh1-zh2",),
        },
        "news-discuss": {
            "lang_pairs": ("en1-en2",),
        },
        "XNLI": {
            "lang_pairs": ("en1-en2", "zh1-zh2"),
        },
        "snli": {
            "lang_pairs": ("en1-en2",),
        },
        "tatoeba": {
            "lang_pairs": ("zh-en", "ita-eng"),
        },
        "convai2": {
            "lang_pairs": ("en1-en2",),
        },
    }

    for folder, params in datasets.items():
        print(f"Processing {folder}")
        inp_dir = _get_inp_dir(folder)
        oup_dir = _get_oup_dir(folder)
        os.makedirs(oup_dir, exist_ok=True)

        for lang_pair in params["lang_pairs"]:
            src, tgt = lang_pair.split("-")
            for l in (src, tgt):
                inp_tmpl = _get_inp_tmpl(folder)
                oup_tmpl = _get_oup_tmpl(folder)
                inp_file = osp.join(inp_dir, inp_tmpl.format(lang_pair, l))
                oup_file = osp.join(oup_dir, oup_tmpl.format(lang_pair, l))
                print(f" - processing {inp_file}")
                process(inp_file, oup_file, _get_lang(l), bpe_codes, args.verbose)


if __name__ == "__main__":
    main()
