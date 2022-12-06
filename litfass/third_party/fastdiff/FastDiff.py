import torch.nn as nn
import torch
import logging
from .module.modules import DiffusionDBlock, TimeAware_LVCBlock
from .module.util import calc_diffusion_step_embedding, std_normal, compute_hyperparams_given_schedule, sampling_given_noise_schedule
from litfass.third_party.argutils import str2bool

def swish(x):
    return x * torch.sigmoid(x)

class FastDiff(nn.Module):
    """FastDiff module."""

    def __init__(
        self,
        audio_channels=1,
        inner_channels=32,
        cond_channels=80,
        upsample_ratios=[8, 8, 4],
        lvc_layers_each_block=4,
        lvc_kernel_size=3,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        dropout=0.0,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        use_weight_norm=True,
        beta_0=1e-6,
        beta_T=0.01,
        T=1000,
    ):
        super().__init__()

        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.audio_channels = audio_channels
        self.cond_channels = cond_channels
        self.lvc_block_nums = len(upsample_ratios)
        self.first_audio_conv = nn.Conv1d(1, inner_channels,
                                    kernel_size=7, padding=(7 - 1) // 2,
                                    dilation=1, bias=True)

        # define residual blocks
        self.lvc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        # the layer-specific fc for noise scale embedding
        self.fc_t = nn.ModuleList()
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        cond_hop_length = 1
        for n in range(self.lvc_block_nums):
            cond_hop_length = cond_hop_length * upsample_ratios[n]
            lvcb = TimeAware_LVCBlock(
                in_channels=inner_channels,
                cond_channels=cond_channels,
                upsample_ratio=upsample_ratios[n],
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
                noise_scale_embed_dim_out=diffusion_step_embed_dim_out
            )
            self.lvc_blocks += [lvcb]
            self.downsample.append(DiffusionDBlock(inner_channels, inner_channels, upsample_ratios[self.lvc_block_nums-n-1]))


        # define output layers
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                inner_channels,
                audio_channels,
                kernel_size=7,
                padding=(7 - 1) // 2,
                dilation=1,
                bias=True
            )
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        self.noise_schedule = torch.linspace(beta_0, beta_T, T)
        self.diffusion_hyperparams = compute_hyperparams_given_schedule(self.noise_schedule)

    def forward(self, x, c, ts=None, reverse=False, mask=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """

        if len(x.shape) == 2:
            B, L = x.shape  # B is batchsize, C=1, L is audio length
            x = x.unsqueeze(1)
        if len(c.shape) == 2:
            c = c.unsqueeze(0)
        B, C, L = c.shape  # B is batchsize, C=80, L is audio length

        if ts is None:
            no_ts = True
            T, alpha = self.diffusion_hyperparams["T"], self.diffusion_hyperparams["alpha"].to(x.device)
            ts = torch.randint(T, size=(B, 1, 1)).to(x.device)  # randomly sample steps from 1~T
            z = std_normal(x.shape, device=x.device)
            delta = (1 - alpha[ts] ** 2.).sqrt()
            alpha_cur = alpha[ts]
            noisy_audio = alpha_cur * x + delta * z  # compute x_t from q(x_t|x_0)
            x = noisy_audio
            ts = ts.view(B, 1)
        else:
            no_ts = False

        # embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(ts, self.diffusion_step_embed_dim_in, device=x.device)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        x = self.first_audio_conv(x)
        downsample = []
        for down_layer in self.downsample:
            downsample.append(x)
            x = down_layer(x)

        for n, audio_down in enumerate(reversed(downsample)):
            x = self.lvc_blocks[n]((x, audio_down, c, diffusion_step_embed))

        # apply final layers
        x = self.final_conv(x)

        if mask is not None:
            x = x.masked_fill(mask, 0)

        if not reverse:
            if no_ts:
                return x, z
            else:
                return x
        else:
            x0 = (noisy_audio - delta * x) / alpha_cur
            return (x, z), x0

    def inference(self, c, N=4, hop_size=256):
        """Inference with the given local conditioning auxiliary features.
        Args:
            c (Tensor): Local conditioning auxiliary features (B, C, T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        c = c.transpose(0, 1)

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
            noise_schedule = torch.FloatTensor(noise_schedule)
        noise_schedule = noise_schedule.to(c.device)

        audio_length = c.shape[-1] * hop_size

        pred_wav = sampling_given_noise_schedule(
            self,
            (1, 1, audio_length),
            self.diffusion_hyperparams,
            noise_schedule,
            condition=c,
            ddim=False,
            return_sequence=False,
            device=c.device
        )

        pred_wav = pred_wav / pred_wav.abs().max()
        pred_wav = pred_wav.view(-1)
        return pred_wav

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments."""
        parser = parent_parser.add_argument_group("FastDiff model setting")
        # network structure related
        parser.add_argument("--fastdiff_audio_channels", default=1, type=int,
                            help="Number of audio channels")
        parser.add_argument("--fastdiff_inner_channels", default=32, type=int,
                            help="Number of inner channels")
        parser.add_argument("--fastdiff_cond_channels", default=80, type=int,
                            help="Number of conditional channels")
        parser.add_argument("--fastdiff_upsample_ratios", default=[8, 8, 4], type=int, nargs="+",
                            help="Upsampling ratios")
        parser.add_argument("--fastdiff_lvc_layers_each_block", default=4, type=int,
                            help="Number of layers in each LVC block")
        parser.add_argument("--fastdiff_lvc_kernel_size", default=3, type=int,
                            help="Kernel size in each LVC block")
        parser.add_argument("--fastdiff_kpnet_hidden_channels", default=64, type=int,
                            help="Number of hidden channels in keypoint network")
        parser.add_argument("--fastdiff_kpnet_conv_size", default=3, type=int,
                            help="Kernel size in keypoint network")
        parser.add_argument("--fastdiff_dropout", default=0.0, type=float,
                            help="Dropout rate")
        parser.add_argument("--fastdiff_diffusion_step_embed_dim_in", default=128, type=int,
                            help="Dimension of diffusion step embedding")
        parser.add_argument("--fastdiff_diffusion_step_embed_dim_mid", default=512, type=int,
                            help="Dimension of diffusion step embedding")
        parser.add_argument("--fastdiff_diffusion_step_embed_dim_out", default=512, type=int,
                            help="Dimension of diffusion step embedding")
        parser.add_argument("--fastdiff_use_weight_norm", default=True, type=str2bool,
                            help="Whether to use weight normalization")
        # training related
        parser.add_argument("--fastdiff_beta_0", default=1e-6, type=float,
                            help="Initial noise scale")
        parser.add_argument("--fastdiff_beta_T", default=0.01, type=float,
                            help="Final noise scale")
        parser.add_argument("--fastdiff_T", default=1000, type=int,
                            help="Number of diffusion steps")
        return parent_parser