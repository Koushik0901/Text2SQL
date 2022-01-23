import wget
import torch
import sentencepiece as spm
from src.model import Txt2SqlTransformer
from src import config


url = "https://github.com/Koushik0901/Text2SQL/releases/download/pretrained-model/txt2sql.pt"
filename = wget.download(url)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = spm.SentencePieceProcessor(model_file="./txt2sql.model")


@torch.no_grad()
def greedy_decode(
    model, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
) -> torch.Tensor:
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            model.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        ).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == config.EOS_ID:
            break
    return ys


def inference(ckpt_path: str, src_sentence: str) -> str:
    ckpt = torch.load(ckpt_path)
    model = Txt2SqlTransformer()
    model.load_state_dict(ckpt["model"])
    model.eval()

    src = torch.tensor(
        [config.BOS_ID] + tokenizer.tokenize(src_sentence) + [config.EOS_ID]
    ).unsqueeze(1)
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,
        src,
        src_mask,
        max_len=40,
        start_symbol=config.BOS_ID,
    ).flatten()
    return tokenizer.decode_ids(tgt_tokens.cpu().tolist())


if __name__ == "__main__":
    print(inference("./txt2sql.pt", "What is Record, when Date is March 1?"))
