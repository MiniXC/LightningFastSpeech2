from torch import nn
import torch


class FastSpeech2Loss(nn.Module):
    def __init__(
        self,
        variances=["energy", "pitch", "snr"],
        variance_levels=["phone", "phone", "phone"],
        duration_type="stochastic",
        max_length=4096,
        loss_alphas={
            "mel": 1.0,
            "pitch": 1e-1,
            "energy": 1e-1,
            "snr": 1e-1,
            "duration": 1e-4,
        },
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.variances = variances
        self.variance_levels = variance_levels
        self.duration_type = duration_type
        self.max_length = max_length
        self.loss_alphas = loss_alphas

    @staticmethod
    def get_loss(pred, truth, loss, mask, unsqueeze=False):
        truth.requires_grad = False
        if unsqueeze:
            mask = mask.unsqueeze(-1)
        pred = pred.masked_select(mask)
        truth = truth.masked_select(mask)
        return loss(pred, truth)

    def forward(self, pred, target, src_mask, tgt_mask):
        variances_pred = {var: pred[var] for var in self.variances}
        variances_target = {var: target[var] for var in self.variances}

        target["duration"] = torch.log(target["duration"].float() + 1)

        src_mask = ~src_mask
        tgt_mask = ~tgt_mask

        losses = {}

        # VARIANCE LOSSES
        if self.max_length is not None:
            assert target["mel"].shape[1] <= self.max_length
            for variance, level in zip(self.variances, self.variance_levels):
                if level == "frame":
                    target[variance] = target[variance][:, : self.max_length]
                    variance_mask = tgt_mask
                elif level == "phone":
                    variance_mask = src_mask
                else:
                    raise ValueError("Unknown variance level: {}".format(level))
                losses[variance] = (
                    FastSpeech2Loss.get_loss(
                        variances_pred[variance],
                        variances_target[variance],
                        self.mse_loss,
                        variance_mask,
                    )
                    * self.loss_alphas[variance]
                )

        # MEL SPECTROGRAM LOSS
        losses["mel"] = (
            FastSpeech2Loss.get_loss(
                pred["mel"], target["mel"], self.l1_loss, tgt_mask, unsqueeze=True
            )
            * self.loss_alphas["mel"]
        )

        # DURATION LOSS
        if self.duration_type == "deterministic":
            losses["duration"] = FastSpeech2Loss.get_loss(
                pred["duration"], target["log_duration"], self.mse_loss, src_mask
            )
        elif self.duration_type == "stochastic":
            losses["duration"] = torch.sum(duration_pred.float())
        losses["duration"] *= self.loss_alphas["duration"]

        # TOTAL LOSS
        losses["total"] = sum(losses.values())

        return losses
