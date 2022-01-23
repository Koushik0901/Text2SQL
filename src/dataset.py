import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import sentencepiece as spm
import config


train_ds = load_dataset("wikisql", split="train")
val_ds = load_dataset("wikisql", split="validation")
test_ds = load_dataset("wikisql", split="test")

tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)


def preprocess_fn(examples):
    ques = examples["question"]
    ans = [x["human_readable"] for x in examples["sql"]]
    return {"input": ques, "target": ans}


def MyCollate(batch):
    questions = [b["input"] for b in batch]
    answers = [b["target"] for b in batch]
    ques = [
        torch.tensor([config.BOS_ID] + x + [config.EOS_ID])
        for x in tokenizer.tokenize(questions)
    ]
    ans = [
        torch.tensor([config.BOS_ID] + x + [config.EOS_ID])
        for x in tokenizer.tokenize(answers)
    ]
    ques_tensor = pad_sequence(ques, padding_value=config.PAD_ID, batch_first=False)
    ans_tensor = pad_sequence(ans, padding_value=config.PAD_ID, batch_first=False)
    return ques_tensor, ans_tensor


def get_dataset():
    encoded_train_ds = train_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["phase", "question", "table", "sql"],
        batch_size=config.BATCH_SIZE,
    )
    encoded_val_ds = val_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["phase", "question", "table", "sql"],
        batch_size=config.BATCH_SIZE,
    )
    encoded_test_ds = test_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["phase", "question", "table", "sql"],
        batch_size=config.BATCH_SIZE,
    )
    return encoded_train_ds, encoded_val_ds, encoded_test_ds
