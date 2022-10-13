from torch import nn
import torch

#from litfass.third_party.softdtw.sdtw_cuda_loss import SoftDTW
from pysdtw import SoftDTW

class FastSpeech2Loss(nn.Module):
    def __init__(
        self,
        variances=["energy", "pitch", "snr"],
        variance_levels=["phone", "phone", "phone"],
        variance_transforms=["cwt", "none", "none"],
        variance_losses=["mse", "mse", "mse"],
        mel_loss="l1",
        duration_loss="mse",
        duration_stochastic=False,
        max_length=4096,
        loss_alphas={
            "mel": 1.0,
            "pitch": 1e-1,
            "energy": 1e-1,
            "snr": 1e-1,
            "duration": 1e-4,
            "gan": 1e-4,
        },
        soft_dtw_gamma=0.01,
        soft_dtw_chunk_size=256,
        variance_gan=False,
    ):
        super().__init__()
        self.losses = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "soft_dtw": SoftDTW(use_cuda=True, gamma=soft_dtw_gamma),
        }
        self.variances = variances
        self.variance_levels = variance_levels
        self.variance_transforms = variance_transforms
        self.variance_losses = variance_losses
        self.duration_stochastic = duration_stochastic
        self.mel_loss = mel_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.duration_loss = duration_loss
        self.max_length = max_length
        self.loss_alphas = loss_alphas
        self.soft_dtw_chunk_size = soft_dtw_chunk_size
        for i, var in enumerate(self.variances):
            if self.variance_transforms[i] == "cwt":
                self.loss_alphas[var + "_cwt"] = self.loss_alphas[var]
                self.loss_alphas[var + "_mean"] = self.loss_alphas[var]
                self.loss_alphas[var + "_std"] = self.loss_alphas[var]
            if variance_gan:
                self.loss_alphas[var + "_discriminator"] = 1 # self.loss_alphas["gan"]
                self.loss_alphas[var + "_generator"] = 1
        self.variance_gan = variance_gan

    def get_loss(self, pred, truth, loss, mask, unsqueeze=False):
        truth.requires_grad = False
        if loss == "soft_dtw" and len(pred.shape) == 2:
            pred = pred.unsqueeze(-1)
            truth = truth.unsqueeze(-1)
        if unsqueeze or loss == "soft_dtw":
            mask = mask.unsqueeze(-1)
        if loss != "soft_dtw":
            pred = pred.masked_select(mask)
            truth = truth.masked_select(mask)
        loss_func = self.losses[loss]
        if loss == "soft_dtw":
            pred = pred.masked_fill_(~mask, 0)
            truth = truth.masked_fill_(~mask, 0)
            pred_chunks = pred.split(self.soft_dtw_chunk_size, dim=1)
            truth_chunks = truth.split(self.soft_dtw_chunk_size, dim=1)
            for i, (pred_chunk, truth_chunk) in enumerate(zip(pred_chunks, truth_chunks)):
                if i == 0:
                    loss = loss_func(pred_chunk, truth_chunk)
                else:
                    loss += loss_func(pred_chunk, truth_chunk)
            loss = loss.sum()
        else:
            loss = loss_func(pred, truth)
        return loss

    def forward(self, result, target, frozen_components=[]):

        losses = {}

        if f"variances_{self.variances[0]}_gen_true" in result:
            variances_gen = {var: result[f"variances_{var}_gen_true"] for var in self.variances}
            for var in self.variances:
                losses[f"{var}_generator"] = self.bce_loss(
                    variances_gen[var], torch.ones_like(variances_gen[var])
                )
            total_loss = sum(
                [
                    v * self.loss_alphas[k]
                    for k, v in losses.items()
                    if not any(f in k for f in frozen_components)
                ]
            )
            losses["total"] = total_loss

            return losses


        variances_pred = {var: result[f"variances_{var}"] for var in self.variances}
        variances_target = {
            var: target[f"variances_{var}"]
            for var in self.variances
            if f"variances_{var}" in target
        }

        src_mask = ~result["src_mask"]
        tgt_mask = ~result["tgt_mask"]

        # DISCRIMINATOR LOSSES
        disc_acc = {}
        if self.variance_gan:
            # true samples
            variances_disc_true = {var: result[f"variances_{var}_disc_true"] for var in self.variances}
            for var in self.variances:
                losses[f"{var}_discriminator"] = self.bce_loss(
                    variances_disc_true[var], torch.ones_like(variances_disc_true[var])
                )
            # fake samples
            variances_disc_fake = {var: result[f"variances_{var}_disc_fake"] for var in self.variances}
            for var in self.variances:
                losses[f"{var}_discriminator"] += self.bce_loss(
                    variances_disc_fake[var], torch.zeros_like(variances_disc_fake[var])
                )
            # accuracy
            for var in self.variances:
                disc_acc[var] = (
                    torch.sum(variances_disc_true[var] > 0.5)
                    + torch.sum(variances_disc_fake[var] < 0.5)
                ) / (variances_disc_true[var].shape[0] + variances_disc_fake[var].shape[0])

        # VARIANCE LOSSES
        if self.max_length is not None:
            assert target["mel"].shape[1] <= self.max_length

            for variance, level, transform, loss in zip(
                self.variances, self.variance_levels, self.variance_transforms, self.variance_losses
            ):
                if transform == "cwt":
                    variances_target[variance] = target[
                        f"variances_{variance}_spectrogram"
                    ]
                    variances_pred[variance] = result[f"variances_{variance}"][
                        "spectrogram"
                    ]
                if level == "frame":
                    if transform != "cwt":
                        variances_target[variance] = variances_target[variance][
                            :, :int(self.max_length)
                        ]
                    variance_mask = tgt_mask
                elif level == "phone":
                    variance_mask = src_mask
                else:
                    raise ValueError("Unknown variance level: {}".format(level))
                if transform == "cwt":
                    losses[variance + "_cwt"] = self.get_loss(
                        variances_pred[variance],
                        variances_target[variance].to(dtype=result["mel"].dtype),
                        loss,
                        variance_mask,
                        unsqueeze=True,
                    )
                    losses[variance + "_mean"] = self.mse_loss(
                        result[f"variances_{variance}"]["mean"],
                        torch.tensor(target[f"variances_{variance}_mean"]).to(
                            result[f"variances_{variance}"]["mean"].device,
                            dtype=result["mel"].dtype,
                        ),
                    )
                    losses[variance + "_std"] = self.mse_loss(
                        result[f"variances_{variance}"]["std"],
                        torch.tensor(target[f"variances_{variance}_std"]).to(
                            result[f"variances_{variance}"]["std"].device,
                            dtype=result["mel"].dtype,
                        ),
                    )
                else:
                    losses[variance] = self.get_loss(
                        variances_pred[variance],
                        variances_target[variance].to(dtype=result["mel"].dtype),
                        loss,
                        variance_mask,
                    )

        # MEL SPECTROGRAM LOSS
        losses["mel"] = self.get_loss(
            result["mel"],
            target["mel"].to(dtype=result["mel"].dtype),
            self.mel_loss,
            tgt_mask,
            unsqueeze=True,
        )

        # DURATION LOSS
        if not self.duration_stochastic:
            losses["duration"] = self.get_loss(
                result["duration_prediction"],
                torch.log(target["duration"] + 1),
                self.duration_loss,
                src_mask,
            )
        else:
            losses["duration"] = torch.sum(result["duration_prediction"])

        # TOTAL LOSS
        total_loss = sum(
            [
                v * self.loss_alphas[k]
                for k, v in losses.items()
                if not any(f in k for f in frozen_components)
            ]
        )
        losses["total"] = total_loss
        if self.variance_gan:
            for var in self.variances:
                losses[f"{var}_discriminator_acc"] = disc_acc[var]

        return losses
