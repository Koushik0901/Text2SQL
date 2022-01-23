import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import Adafactor
from torch.utils.data import DataLoader
from model import Txt2SqlTransformer
from dataset import get_dataset, MyCollate
from typing import Tuple
import config


class LitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        vocab_size: int = config.VOCAB_SIZE,
        embed_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 16,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.model = Txt2SqlTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            d_model=d_model,
            num_layers=num_layers,
            n_head=n_heads,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

        self.lr = lr

    def forward(
        self,
        src: torch.Tensor,
        trg_input: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        trg_padding_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            src,
            trg_input,
            src_mask,
            trg_mask,
            src_padding_mask,
            trg_padding_mask,
            memory_mask,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        src, trg = batch
        trg_input = trg[:-1, :]
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.model.create_mask(
            src, trg_input
        )
        logits = self.model(
            src,
            trg_input,
            src_mask.to(self.device),
            trg_mask.to(self.device),
            src_padding_mask.to(self.device),
            trg_padding_mask.to(self.device),
            src_padding_mask.to(self.device),
        )
        trg_out = trg[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        src, trg = batch
        trg_input = trg[:-1, :]
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.model.create_mask(
            src, trg_input
        )
        logits = self.model(
            src,
            trg_input,
            src_mask.to(self.device),
            trg_mask.to(self.device),
            src_padding_mask.to(self.device),
            trg_padding_mask.to(self.device),
            src_padding_mask.to(self.device),
        )
        trg_out = trg[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        src, trg = batch
        trg_input = trg[:-1, :]
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.model.create_mask(
            src, trg_input
        )
        logits = self.model(
            src,
            trg_input,
            src_mask.to(self.device),
            trg_mask.to(self.device),
            src_padding_mask.to(self.device),
            trg_padding_mask.to(self.device),
            src_padding_mask.to(self.device),
        )
        trg_out = trg[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adafactor(
            self.model.parameters(),
            lr=self.lr,
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False,
        )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            collate_fn=MyCollate,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            eval_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=MyCollate,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=MyCollate,
        )
        return test_loader


if __name__ == "__main__":
    train_ds, eval_ds, test_ds = get_dataset()
    model = LitModel()
    swa = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=5, device="cuda")
    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath="./",
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=True,
        save_top_k=1,
    )
    wandb_logger = WandbLogger(project="text2sql", log_model=True)
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[checkpointer, swa],
        logger=wandb_logger,
        accumulate_grad_batches=5,
        max_epochs=50,
    )

    # learning rate finder
    lr_finder = trainer.tuner.lr_find(model)
    new_lr = lr_finder.suggestion()
    model.lr = new_lr
    print("new lr", new_lr)

    trainer.fit(model)
    trainer.test(model)
