from litfass.fastspeech2.model import VarianceConvolutionLayer
# from litfass.third_party.fastdiff.module.utils

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
        use_weight_norm=True,
        beta_0=1e-6,
        beta_T=0.01,
        T=1000,
    ):
        super().__init__()

        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t = nn.ModuleList(
            nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid),
            nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(
                inner_channels,
                1,
                kernel_size=7,
                padding=(7 - 1) // 2,
                dilation=1,
                bias=True
            )
        )