import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("wikisql", split="train")

with open("tokenizer_train.txt", "a") as f:
    for i in tqdm(range(len(dataset))):
        row = dataset[i]
        f.write(row["question"] + "\n")
        f.write(row["sql"]["human_readable"] + "\n")

spm.SentencePieceTrainer.train(
    input="tokenizer_train.txt",
    model_prefix="txt2sql",
    model_type="bpe",
    vocab_size=20000,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
