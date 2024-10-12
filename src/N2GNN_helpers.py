from argparse import ArgumentParser
from copy import deepcopy as c
from typing import Dict, Optional

from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric
from pathlib import Path
import sys
sys.path.insert(1, Path("../N2GNN").resolve().as_posix())

from models.model_construction import make_model


class PlGNNModule(LightningModule):
    r"""Basic pytorch lighting module for GNNs.
    Args:
        loss_criterion (nn.Module) : Loss compute module.
        evaluator (Metric): Evaluator for evaluating model performance.
        target_variable (str): Target variable name in the dataset for model prediction
        loss_criterion (nn.Module): Loss compute module.
        args (ArgumentParser): Arguments dict from argparser.
        init_encoder (nn.Module, optional): Node feature initial encoder.
        edge_encoder (nn.Module, optional): Edge feature encoder.
    """

    def __init__(self,
                target_variable: str,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 args: ArgumentParser,
                 init_encoder: Optional[nn.Module] = None,
                 edge_encoder: Optional[nn.Module] = None,
                 ):
        super(PlGNNModule, self).__init__()
        self.model = make_model(args, init_encoder, edge_encoder)
        self.target_variable = target_variable
        self.loss_criterion = loss_criterion
        self.train_evaluator = c(evaluator)
        self.val_evaluator = c(evaluator)
        self.test_evaluator = c(evaluator)
        self.args = args
        self.classification = args.dataset_name == "classification"

    def forward(self, data: Data) -> Tensor:
        if self.classification:
            return torch.sigmoid(self.model(data))[:, 0]
        return self.model(data)

    def training_step(self, batch: Data, batch_idx: Tensor) -> Dict:
        y = getattr(batch, self.target_variable)
        out = self.forward(batch)
        loss = self.loss_criterion(out, y)
        self.log("train/loss",
                 loss,
                 prog_bar=True,
                 batch_size=self.args.batch_size)
        self.train_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_train_epoch_end(self):
        self.log("train/metric",
                 self.train_evaluator.compute(),
                 prog_bar=False)
        self.train_evaluator.reset()

    def validation_step(self, batch: Data, batch_idx: Tensor) -> Dict:
        y = getattr(batch, self.target_variable)
        out = self.forward(batch)
        loss = self.loss_criterion(out, y)
        self.log("val/loss",
                 loss,
                 prog_bar=False,
                 batch_size=self.args.batch_size)
        self.val_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_validation_epoch_end(self):
        self.log("val/metric",
                 self.val_evaluator.compute(),
                 prog_bar=True)
        self.val_evaluator.reset()

    def test_step(self, batch: Data, batch_idx: Tensor) -> Dict:
        y = getattr(batch, self.target_variable)
        out = self.forward(batch)
        loss = self.loss_criterion(out, y)
        self.log("test/loss",
                 loss,
                 prog_bar=False,
                 batch_size=self.args.batch_size)
        self.test_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y}

    def on_test_epoch_end(self) -> None:
        self.log("test/metric",
                 self.test_evaluator.compute(),
                 prog_bar=True)
        self.test_evaluator.reset()

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=float(self.args.l2_wd),
        )
        if self.args.lr_scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=self.args.mode, factor=self.args.factor, patience=self.args.lr_patience, min_lr=self.args.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler":  scheduler,
                "monitor": "val/metric",
                "frequency": 1,
                "interval": "epoch",
            },
        }

    def get_progress_bar_dict(self) -> Dict:
        r"""Remove 'v_num' in progress bar for clarity"""
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class PlGNNTestonValModule(PlGNNModule):
    r"""Given a preset evaluation interval, run test dataset during validation
        to have a snoop on test performance every args.test_eval_interval epochs during training.
    """

    def __init__(self,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 args: ArgumentParser,
                 init_encoder: Optional[nn.Module] = None,
                 edge_encoder: Optional[nn.Module] = None,
                 ):
        super(PlGNNTestonValModule, self).__init__(loss_criterion, evaluator, args, init_encoder, edge_encoder)
        self.test_eval_still = self.args.test_eval_interval

    def validation_step(self,
                        batch: Data,
                        batch_idx: Tensor,
                        dataloader_idx: int) -> Dict:

        if dataloader_idx == 0:
            y = getattr(batch, self.target_variable)
            out = self.forward(batch)
            loss = self.loss_criterion(out, y)
            self.log("val/loss",
                     loss,
                     prog_bar=False,
                     batch_size=self.args.batch_size,
                     add_dataloader_idx=False)
            self.val_evaluator.update(out, y)
        else:
            if self.test_eval_still != 0:
                return {'loader_idx': dataloader_idx}
            # only do validation on test set when reaching the predefined epoch.
            else:
                y = getattr(batch, self.target_variable)
                out = self.forward(batch)
                loss = self.loss_criterion(out, y)
                self.log("test/loss",
                         loss,
                         prog_bar=False,
                         batch_size=self.args.batch_size,
                         add_dataloader_idx=False)
                self.test_evaluator.update(out, y)
        return {'loss': loss, 'preds': out, 'target': y, 'loader_idx': dataloader_idx}

    def on_validation_epoch_end(self):
        self.log("val/metric",
                 self.val_evaluator.compute(),
                 prog_bar=True,
                 add_dataloader_idx=False)
        self.val_evaluator.reset()
        if self.test_eval_still == 0:
            self.log("test/metric",
                     self.test_evaluator.compute(),
                     prog_bar=True,
                     add_dataloader_idx=False)
            self.test_evaluator.reset()
            self.test_eval_still = self.args.test_eval_interval
        else:
            self.test_eval_still = self.test_eval_still - 1

    def set_test_eval_still(self):
        # set test validation interval to zero to performance test dataset validation.
        self.test_eval_still = 0

    def on_test_epoch_start(self):
        self.set_test_eval_still()

    def test_step(self,
                  batch: Data,
                  batch_idx: Tensor,
                  dataloader_idx: int) -> Dict:
        results = self.validation_step(batch, batch_idx, dataloader_idx)
        return results

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
        

class PlGNNTestonValModule(PlGNNModule):
    r"""Given a preset evaluation interval, run test dataset during validation
        to have a snoop on test performance every args.test_eval_interval epochs during training.
    """

    def __init__(self,
                 target_variable: str,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 args: ArgumentParser,
                 init_encoder: Optional[nn.Module] = None,
                 edge_encoder: Optional[nn.Module] = None,
                 is_classification: bool = False,
                 ):
        super(PlGNNTestonValModule, self).__init__(target_variable, loss_criterion, evaluator, args, init_encoder, edge_encoder)
        self.target_variable = target_variable
        self.is_classification = is_classification
        self.test_eval_still = self.args.test_eval_interval

    def validation_step(self,
                        batch: Data,
                        batch_idx: Tensor,
                        dataloader_idx: int) -> Dict:

        if dataloader_idx == 0:
            y = getattr(batch, self.target_variable)
            out = self.forward(batch)
            loss = self.loss_criterion(out, y)
            self.log("val/loss",
                     loss,
                     prog_bar=False,
                     batch_size=self.args.batch_size,
                     add_dataloader_idx=False)
            if self.is_classification:
                accuracy = ((out > 0.5) == y).float().mean()
                self.log("val/accuracy",
                         accuracy,
                         prog_bar=False,
                         batch_size=self.args.batch_size,
                         add_dataloader_idx=False,
                         on_step=False,)
            self.val_evaluator.update(out, y)
        else:
            if self.test_eval_still != 0:
                return {'loader_idx': dataloader_idx}
            # only do validation on test set when reaching the predefined epoch.
            y = getattr(batch, self.target_variable)
            out = self.forward(batch)
            loss = self.loss_criterion(out, y)
            self.log("test/loss",
                        loss,
                        prog_bar=False,
                        batch_size=self.args.batch_size,
                        add_dataloader_idx=False)
            if self.is_classification:
                accuracy = ((out > 0.5) == y).float().mean()
                self.log("test/accuracy",
                         accuracy,
                         prog_bar=False,
                         batch_size=self.args.batch_size,
                         add_dataloader_idx=False,
                         on_step=False,)
            self.test_evaluator.update(out, y)

        return {'loss': loss, 'preds': out, 'target': y, 'loader_idx': dataloader_idx}

    def on_validation_epoch_end(self):
        val_metric = self.val_evaluator.compute()
        self.log("val/metric",
                 val_metric,
                 prog_bar=True,
                 add_dataloader_idx=False)
        
                # Log the minimum validation error over the entire training process
        if self.current_epoch == 0:
            self.min_val_loss = val_metric
        else:
            self.min_val_loss = min(self.min_val_loss, val_metric)
        self.log("val/min_loss",
                 self.min_val_loss,
                 prog_bar=True,
                 add_dataloader_idx=False)
            
        self.val_evaluator.reset()
        if self.test_eval_still == 0:
            self.log("test/metric",
                     self.test_evaluator.compute(),
                     prog_bar=True,
                     add_dataloader_idx=False)
            self.test_evaluator.reset()
            self.test_eval_still = self.args.test_eval_interval
        else:
            self.test_eval_still = self.test_eval_still - 1
            


    def set_test_eval_still(self):
        # set test validation interval to zero to performance test dataset validation.
        self.test_eval_still = 0

    def on_test_epoch_start(self):
        self.set_test_eval_still()

    def test_step(self,
                  batch: Data,
                  batch_idx: Tensor,
                  dataloader_idx: int) -> Dict:
        results = self.validation_step(batch, batch_idx, dataloader_idx)
        return results

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()