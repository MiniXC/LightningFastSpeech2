from torch import nn
import torch

class FastSpeech2Loss(nn.Module):
    def __init__(
        self,
        variances=["energy", "pitch", "snr"],
        variance_levels=["phone", "phone", "phone"],
        variance_transforms=["cwt", "none", "none"],
        duration_stochastic=False,
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
        self.variance_transforms = variance_transforms
        self.duration_stochastic = duration_stochastic
        self.max_length = max_length
        self.loss_alphas = loss_alphas

    @staticmethod
    def get_loss(pred, truth, loss, mask, unsqueeze=False, cwt=False):
        truth.requires_grad = False
        if unsqueeze:
            mask = mask.unsqueeze(-1)
        if cwt:
            mask = torch.stack([mask] * 10, dim=-1)
        pred = pred.masked_select(mask)
        truth = truth.masked_select(mask)
        return loss(pred, truth)

    def forward(self, result, target):
        variances_pred = {var: result[f"variances_{var}"] for var in self.variances}
        variances_target = {var: target[f"variances_{var}"] for var in self.variances if f"variances_{var}" in target}

        src_mask = ~result["src_mask"]
        tgt_mask = ~result["tgt_mask"]

        losses = {}

        # VARIANCE LOSSES
        if self.max_length is not None:
            assert target["mel"].shape[1] <= self.max_length
            
            for variance, level, transform in zip(self.variances, self.variance_levels, self.variance_transforms):
                if transform == "cwt":
                    variances_target[variance] = target[f"variances_{variance}_spectrogram"]
                    variances_pred[variance] = result[f"variances_{variance}"]["spectrogram"]
                if level == "frame":
                    if transform != "cwt":
                        variances_target[variance] = variances_target[variance][:, :int(self.max_length)]
                    variance_mask = tgt_mask
                elif level == "phone":
                    variance_mask = src_mask
                else:
                    raise ValueError("Unknown variance level: {}".format(level))
                if transform == "cwt":
                    losses[variance] = (
                        FastSpeech2Loss.get_loss(
                            variances_pred[variance].float(),
                            variances_target[variance].float(),
                            self.l1_loss,
                            variance_mask,
                            cwt=True,
                        )
                        + self.mse_loss(result[f"variances_{variance}"]["mean"].float(), torch.tensor(target[f"variances_{variance}_mean"]).to(result[f"variances_{variance}"]["mean"].device).float())
                        + self.mse_loss(result[f"variances_{variance}"]["std"].float(), torch.tensor(target[f"variances_{variance}_std"]).to(result[f"variances_{variance}"]["std"].device).float())
                    )
                else:
                    losses[variance] = (
                        FastSpeech2Loss.get_loss(
                            variances_pred[variance].float(),
                            variances_target[variance].float(),
                            self.mse_loss,
                            variance_mask,
                        )
                        * self.loss_alphas[variance]
                    )

        # MEL SPECTROGRAM LOSS
        losses["mel"] = (
            FastSpeech2Loss.get_loss(
                result["mel"].float(), target["mel"].float(), self.l1_loss, tgt_mask, unsqueeze=True
            )
            * self.loss_alphas["mel"]
        )

        # DURATION LOSS
        if not self.duration_stochastic:
            losses["duration"] = FastSpeech2Loss.get_loss(
                result["duration_prediction"], torch.log(target["duration"] + 1), self.mse_loss, src_mask
            )
        else:
            losses["duration"] = torch.sum(result["duration_prediction"].float())
        losses["duration"] *= self.loss_alphas["duration"]

        # TOTAL LOSS
        losses["total"] = sum(losses.values())

        return losses
