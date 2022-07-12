import torch

from third_party.hifigan import Synthesiser
from fastspeech2.fastspeech2 import FastSpeech2
from .g2p import G2P


class SpeechGenerator:
    def __init__(self, model_path: str, g2p_model: G2P):
        self.model_path = model_path
        self.synth = Synthesiser()
        self.model = FastSpeech2.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.g2p = g2p_model

    @property
    def speakers(self):
        if self.model.hparams.speaker_type == "dvector":
            return self.model.speaker2dvector.keys()
        elif self.model.hparams.speaker_type == "id":
            return self.model.speaker2id.keys()
        else:
            return None

    def generate_sample_from_text(self, text, speaker=None):
        if self.model.hparams.speaker_type != "none" and speaker is not None:
            raise Exception("Speaker is required for this model")
        ids = [self.model.phone2id[x] for x in self.g2p(text)]
        batch = {}
        if self.model.hparams.speaker_type == "dvector":
            batch["speaker"] = torch.tensor([self.model.speaker2dvector[speaker]])
        if self.model.hparams.speaker_type == "id":
            batch["speaker"] = torch.tensor([self.model.speaker2id[speaker]])
        batch["phones"] = torch.tensor([ids])
        result = self.model(batch, inferece=True)
        pred_mel = result["mel"][0][~result["tgt_mask"][0]].cpu()
        return self.synth(pred_mel.to("cuda:0"))[0]
