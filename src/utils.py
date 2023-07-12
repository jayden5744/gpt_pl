import os
import os.path as osp
import shutil

import sentencepiece as spm


def exist_file(path: str) -> bool:
    if osp.exists(path):
        return True
    return False


def create_or_load_tokenizer(
    file_path: str,
    save_path: str,
    language: str,
    vocab_size: int,
    tokenizer_type: str = "bpe",
    bos_token: str = "[BOS]",
    eos_token: str = "[EOS]",
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
) -> spm.SentencePieceProcessor:
    corpus_prefix = f"{language}_corpus_{vocab_size}"

    if tokenizer_type.strip().lower() not in ["unigram", "bpe", "char", "word"]:
        raise ValueError(
            f"param `tokenizer_type` must be one of [unigram, bpe, char, word]"
        )

    if not os.path.isdir(save_path):  # 폴더 없으면 만들어
        os.makedirs(save_path)

    model_path = osp.join(save_path, corpus_prefix + ".model")
    vocab_path = osp.join(save_path, corpus_prefix + ".vocab")

    if not exist_file(model_path) and not exist_file(vocab_path):
        model_train_cmd = f"--input={file_path} --model_prefix={corpus_prefix} --model_type={tokenizer_type} --vocab_size={vocab_size}  --bos_piece={bos_token}  --eos_piece={eos_token}  --unk_piece={unk_token} --pad_piece={pad_token}"
        spm.SentencePieceTrainer.Train(model_train_cmd)
        shutil.move(corpus_prefix + ".model", model_path)
        shutil.move(corpus_prefix + ".vocab", vocab_path)
    # model file은 있는데, vocab file이 없거나 / model_file은 없는데, vocab file이 있으면 -> Error

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
