from torch import nn
import torch
from litfass.fastspeech2.model import VarianceConvolutionLayer, LengthRegulator
from litfass.third_party.fastdiff.module.util import calc_diffusion_step_embedding, std_normal, compute_hyperparams_given_schedule, sampling_given_noise_schedule
from litfass.third_party.fastdiff.FastDiff import swish


class FastDiffVarianceAdaptor(nn.Module):
    """FastSpeech2 Variance Adaptor with FastDiff diffusion. Limited to 1d, frame-level predictions."""
    def __init__(
        self,
        stats,
        variances,
        variance_nlayers,
        variance_kernel_size,
        variance_dropout,
        variance_filter_size,
        variance_nbins,
        variance_depthwise_conv,
        duration_nlayers,
        duration_kernel_size,
        duration_dropout,
        duration_filter_size,
        duration_depthwise_conv,
        encoder_hidden,
        max_length,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        beta_0=1e-6,
        beta_T=0.01,
        T=1000,
    ):
        super().__init__()

        self.max_length = max_length
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.noise_schedule = torch.linspace(beta_0, beta_T, T)
        self.diffusion_hyperparams = compute_hyperparams_given_schedule(self.noise_schedule)

        self.duration_predictor = FastDiffVariancePredictor(
            duration_nlayers,
            encoder_hidden,
            duration_filter_size,
            duration_kernel_size,
            duration_dropout,
            duration_depthwise_conv,
            self.diffusion_hyperparams,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
        )

        self.length_regulator = LengthRegulator(pad_to_multiple_of=64) # TODO: change this to use target length

        self.variances = variances

        self.encoders = {}
        for var in self.variances:
            self.encoders[var] = FastDiffVarianceEncoder(
                variance_nlayers[variances.index(var)],
                encoder_hidden,
                variance_filter_size,
                variance_kernel_size[variances.index(var)],
                variance_dropout[variances.index(var)],
                variance_depthwise_conv,
                stats[var]["min"],
                stats[var]["max"],
                stats[var]["mean"],
                stats[var]["std"],
                variance_nbins,
                self.diffusion_hyperparams,
                diffusion_step_embed_dim_in,
                diffusion_step_embed_dim_mid,
                diffusion_step_embed_dim_out,
            )
        self.encoders = nn.ModuleDict(self.encoders)


    def forward(
        self,
        x,
        src_mask,
        targets,
        inference=False,
        N=4,
    ):
        if not inference:
            duration = targets["duration"] + 1 + torch.rand(size=targets["duration"].shape, device=targets["duration"].device)*0.49
            duration = (torch.log(duration) - 1.08) / 0.96
            duration_pred, duration_z = self.duration_predictor(
                duration.to(x.dtype),
                x.transpose(1, 2), 
                mask=src_mask
            )
        else:
            duration_pred = self.duration_predictor.inference(x, N=N)
            duration_z = None

        result = {}

        out_val = None

        if not inference:
            duration_rounded = targets["duration"]
        else:
            duration_pred = duration_pred * 0.96 + 1.08
            duration_rounded = torch.round((torch.exp(duration_pred) - 1))
            duration_rounded = torch.clamp(duration_rounded, min=0).int()
            for i in range(len(duration_rounded)):
                if duration_rounded[i][~src_mask[i]].sum() <= (~src_mask[i]).sum() // 2:
                    duration_rounded[i][~src_mask[i]] = 1
                    print("Zero duration, setting to 1")
                duration_rounded[i][src_mask[i]] = 0

        x, tgt_mask = self.length_regulator(x, duration_rounded, self.max_length)
        if out_val is not None:
            out_val, _ = self.length_regulator(out_val, duration_rounded, self.max_length)

        for i, var in enumerate(self.variances):
            if not inference:
                (pred, z), out = self.encoders[var](
                    x.transpose(1, 2), targets[f"variances_{var}"], tgt_mask
                )
            else:
                pred, out = self.encoders[var](x, None, tgt_mask)
                z = None
            result[f"variances_{var}"] = pred
            result[f"variances_{var}_z"] = z
            if out_val is None:
                out_val = out
            else:
                out_val = out_val + out
                x = x + out

        result["x"] = x
        result["duration_prediction"] = duration_pred
        result["duration_z"] = duration_z
        result["duration_rounded"] = duration_rounded
        result["tgt_mask"] = tgt_mask
        result["out"] = out_val

        return result


class FastDiffVariancePredictor(nn.Module):
    def __init__(
        self,
        nlayers,
        in_channels,
        filter_size,
        kernel_size,
        dropout,
        depthwise,
        diffusion_hyperparams,
        diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out,
    ):
        super().__init__()

        self.diffusion_hyperparams = diffusion_hyperparams

        self.linear_in = nn.Linear(1, in_channels)

        self.layers = nn.Sequential(
            *[
                VarianceConvolutionLayer(
                    in_channels, filter_size, kernel_size, dropout, depthwise
                )
                for _ in range(nlayers)
            ]
        )

        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t = nn.ModuleList()
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.linear = nn.Linear(filter_size, 1)

        self.linear_noise = nn.Linear(diffusion_step_embed_dim_out, in_channels)

    def forward(self, x, c, ts=None, mask=None):
        # print(x.shape, c.shape, "input shapes")

        if len(x.shape) == 2:
            B, L = x.shape  # B is batchsize, C=1, L is audio length
            x = x.unsqueeze(1)
        if len(c.shape) == 2:
            c = c.unsqueeze(0)

        B, C, L = c.shape

        if ts is None:
            no_ts = True
            T, alpha = self.diffusion_hyperparams["T"], self.diffusion_hyperparams["alpha"].to(x.device)
            ts = torch.randint(T, size=(B, 1, 1)).to(x.device)  # randomly sample steps from 1~T
            z = std_normal(x.shape, device=x.device).to(x.dtype)
            delta = (1 - alpha[ts] ** 2.).sqrt()
            alpha_cur = alpha[ts]
            noisy_audio = alpha_cur * x + delta * z  # compute x_t from q(x_t|x_0)
            x = noisy_audio
            ts = ts.view(B, 1)
        else:
            no_ts = False

        # embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(ts, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        x = x.transpose(1,2)
        x = self.linear_in(x.to(diffusion_step_embed.dtype))
        x = x.transpose(1,2)
        c = c.to(diffusion_step_embed.dtype)
        noise_embed = self.linear_noise(diffusion_step_embed).unsqueeze(1).transpose(1, 2)

        # print(x.shape, c.shape, noise_embed.shape, "forward shapes")
        # print(x.dtype, c.dtype, diffusion_step_embed.dtype)

        out_conv = self.layers(
            (x+c+noise_embed).transpose(1, 2)
        )
        out = self.linear(out_conv)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0)
        
        if no_ts:
            return out, z
        else:
            return out

    def inference(self, c, N=4):
        """Inference with the given local conditioning auxiliary features.
        Args:
            c (Tensor): Local conditioning auxiliary features (B, C, T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        c = c.transpose(1, 2)

        reverse_step = N
        if reverse_step == 1000:
                noise_schedule = torch.linspace(0.000001, 0.01, 1000)
        elif reverse_step == 200:
            noise_schedule = torch.linspace(0.0001, 0.02, 200)
        elif reverse_step == 8:
            noise_schedule = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5]
        elif reverse_step == 6:
            noise_schedule = [1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
                                0.006634317338466644, 0.09357017278671265, 0.6000000238418579]
        elif reverse_step == 4:
            noise_schedule = [3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01]
        elif reverse_step == 3:
            noise_schedule = [9.0000e-05, 9.0000e-03, 6.0000e-01]
        else:
            raise ValueError("Reverse step should be 3, 4, 6, 8, 200 or 1000.")

        if not isinstance(noise_schedule, torch.Tensor):
            noise_schedule = torch.FloatTensor(noise_schedule).to(c.dtype)
        noise_schedule = noise_schedule.to(c.device)

        audio_length = c.shape[-1]

        # print(c.shape, "c shape, inference")

        pred_wav = sampling_given_noise_schedule(
            self,
            (c.shape[0], audio_length),
            self.diffusion_hyperparams,
            noise_schedule,
            condition=c,
            ddim=False,
            return_sequence=False,
            device=c.device,
        )

        # pred_wav = pred_wav / pred_wav.abs().max(axis=1, keepdim=True)[0]
        # pred_wav = pred_wav.view(-1)
        return pred_wav

class FastDiffVarianceEncoder(nn.Module):
    def __init__(
        self,
        nlayers,
        in_channels,
        filter_size,
        kernel_size,
        dropout,
        depthwise,
        min,
        max,
        mean,
        std,
        nbins,
        diffusion_hyperparams,
        diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out,
    ):
        super().__init__()
        self.predictor = FastDiffVariancePredictor(
            nlayers, 
            in_channels,
            filter_size,
            kernel_size,
            dropout,
            depthwise,
            diffusion_hyperparams,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
        )
        self.bins = nn.Parameter(
            torch.linspace(min, max, nbins - 1),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(nbins, in_channels)
        self.mean = mean
        self.std = std

    def forward(self, x, tgt, mask, N=4, control=1.0):
        if tgt is not None:
            # training
            noise_pred, z = self.predictor(tgt, x, mask=mask)
            tgt = tgt * self.std + self.mean
            embedding = self.embedding(torch.bucketize(tgt, self.bins).to(x.device))
            return (noise_pred, z), embedding
        else:
            # inference
            prediction = self.predictor.inference(x, N=N)
            bucket_prediction = prediction * self.std + self.mean
            prediction = prediction * control
            embedding = self.embedding(
                torch.bucketize(bucket_prediction, self.bins).to(x.device)
            )
            return prediction, embedding

class FastDiffSpeakerGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim,
        c_dim,
        speaker_embed_dim,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        beta_0=1e-6,
        beta_T=0.01,
        T=1000,
    ):
        super().__init__()
        self.noise_schedule = torch.linspace(beta_0, beta_T, T)
        self.diffusion_hyperparams = compute_hyperparams_given_schedule(self.noise_schedule)
        
        self.predictor = FastDiffSpeakerPredictor(
            hidden_dim,
            c_dim,
            speaker_embed_dim,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            beta_0,
            beta_T,
            T,
        )

    def forward(
        self,
        x,
        dvec=None,
        inference=False,
        N=4,
    ):
        if inference:
            # inference
            prediction = self.predictor.inference(x, N=N)
            return prediction
        else:
            # training
            noise_pred, z = self.predictor(dvec, x)
            return noise_pred, z


class FastDiffSpeakerPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        c_dim,
        speaker_embed_dim,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        beta_0=1e-6,
        beta_T=0.01,
        T=1000,
    ):
        super().__init__()

        self.noise_schedule = torch.linspace(beta_0, beta_T, T)
        diffusion_hyperparams = compute_hyperparams_given_schedule(self.noise_schedule)

        self.diffusion_hyperparams = diffusion_hyperparams

        self.mlp = nn.Sequential(
            *[
                nn.Linear(speaker_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        )

        self.conditional_in = nn.Linear(c_dim, speaker_embed_dim)
        self.linear_out = nn.Linear(hidden_dim, speaker_embed_dim)

        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t = nn.ModuleList()
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.linear_noise = nn.Linear(diffusion_step_embed_dim_out, speaker_embed_dim)

    def forward(self, x, c, ts=None, mask=None):
        # print(x.shape, c.shape, "input shapes")

        B, C = c.shape

        if ts is None:
            no_ts = True
            T, alpha = self.diffusion_hyperparams["T"], self.diffusion_hyperparams["alpha"].to(x.device)
            ts = torch.randint(T, size=(B, 1, 1)).to(x.device)  # randomly sample steps from 1~T
            x = x.unsqueeze(-1)
            z = std_normal(x.shape, device=x.device).to(x.dtype)
            delta = (1 - alpha[ts] ** 2.).sqrt()
            alpha_cur = alpha[ts]
            noisy_audio = alpha_cur * x + delta * z  # compute x_t from q(x_t|x_0)
            x = noisy_audio
            x = x.squeeze(-1)
            z = z.squeeze(-1)
            ts = ts.view(B, 1)
        else:
            no_ts = False

        # embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(ts, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # x = x.transpose(1,2)
        # x = self.mlp(x.to(diffusion_step_embed.dtype))
        # x = x.transpose(1,2)
        c = self.conditional_in(c.to(diffusion_step_embed.dtype)) # TODO: investigate attention here
        noise_embed = self.linear_noise(diffusion_step_embed)

        # print(x.shape, c.shape, noise_embed.shape, "forward shapes")
        # print(x.dtype, c.dtype, diffusion_step_embed.dtype)

        out_conv = self.mlp(
            (x+c+noise_embed)
        )
        out = self.linear_out(out_conv)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0)
        
        if no_ts:
            return out, z
        else:
            return out

    def inference(self, c, N=4):
        """Inference with the given local conditioning auxiliary features.
        Args:
            c (Tensor): Local conditioning auxiliary features (B, C, T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """

        reverse_step = N
        if reverse_step == 1000:
                noise_schedule = torch.linspace(0.000001, 0.01, 1000)
        elif reverse_step == 200:
            noise_schedule = torch.linspace(0.0001, 0.02, 200)
        elif reverse_step == 8:
            noise_schedule = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5]
        elif reverse_step == 6:
            noise_schedule = [1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
                                0.006634317338466644, 0.09357017278671265, 0.6000000238418579]
        elif reverse_step == 4:
            noise_schedule = [3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01]
        elif reverse_step == 3:
            noise_schedule = [9.0000e-05, 9.0000e-03, 6.0000e-01]
        else:
            raise ValueError("Reverse step should be 3, 4, 6, 8, 200 or 1000.")

        if not isinstance(noise_schedule, torch.Tensor):
            noise_schedule = torch.FloatTensor(noise_schedule).to(c.dtype)
        noise_schedule = noise_schedule.to(c.device)

        audio_length = c.shape[-1]

        # print(c.shape, "c shape, inference")

        pred_wav = sampling_given_noise_schedule(
            self,
            (c.shape[0], audio_length),
            self.diffusion_hyperparams,
            noise_schedule,
            condition=c,
            ddim=False,
            return_sequence=False,
            device=c.device
        )

        # pred_wav = pred_wav / pred_wav.abs().max(axis=1, keepdim=True)[0]
        # pred_wav = pred_wav.view(-1)
        return pred_wav