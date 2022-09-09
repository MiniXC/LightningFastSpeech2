# LightningFastSpeech

**WARNING: This is a work in progress and until version 0.1 (which will be out very soon), it might be hard to get running on your own machine. Thanks for your patience.**

## Large Pretrained TTS

In the NLP community, and more recently in speech recognition, large pre-trained models and how they can be used for down-stream tasks have become an exciting area of research.

In TTS however, little similar work exists. With this project, I hope to make a first step into bringing pretrained models to TTS.
The original FastSpeech 2 model is 27M parameters large and models a single speaker, while our version would have almost 2B parameters without the improvements from LightSpeech, which bring its size down to a manageable 135M, and models more than 2,000 speakers.

A big upside of this implementation is that it is based on [Pytorch Lightning](https://www.pytorchlightning.ai/), which makes it easy to do multi-gpu training, load pre-trained models and a lot more.

LightningFastSpeech couldn't exist without the amazing open source work of many others, for a full list see [Attribution](#attribution).

## Current Status

This library is a work in progress, and until v1.0, updates might break things occasionally.

# Goals

## v0.1

**0.1** is right around the corner! For this version, the core functionality is already there, and what's missing are mostly quality of life improvements that we should get out of the way now.

- [x] Replicate original FastSpeech 2 architecture
- [x] Include Depth-wise separable convolutions found in LightSpeech
- [x] Dataloader which computes prosody features online
- [x] Synthesis of both individual utterances and whole datasets
- [x] Configurable training script.
- [ ] Configurable synthesis script.
- [ ] First large pre-trained model (LibriTTS, 2k speakers, 135M).
- [ ] Documentation & tutorials.
- [ ] Configurable metrics.
- [ ] LJSpeech support.
- [ ] PyPi package.

## v1.0

It will take a while to get to 1.0 -- the goal for this to allow everyone to easily fine-tune our models and to easily do controllable synthesis of utterances.

- [ ] Allow models to be loaded from the [Huggingface hub](huggingface.co/models).
- [ ] [Streamlit](https://streamlit.io/) interface for synthesising utterances and generating datasets.
- [ ] [Tract](https://github.com/sonos/tract) and [tractjs](https://bminixhofer.github.io/tractjs/) integration to export models for on-device and web use.
- [ ] Make it easy to add new datasets and to fine-tune models with them.
- [ ] Add HiFi-GAN fine-tuning to the pipeline.
- [ ] A range of pre-trained models with different domains and sizes (e.g. multi-lingual, noisy/clean)

# Attribution

This would not be possible without a lot of amazing open source project in the TTS space already present -- please cite their work when appropriate!

- [Chung-Ming Chien's FastSpeech 2 implementation](https://github.com/ming024/FastSpeech2), which was used during as a reference implementation.
- [yistLin's public d-vector implementation](https://github.com/yistLin/dvector), which is used for multi-speaker training.
- [Aidan Pine's fork of FastSpeech 2](https://github.com/roedoejet/FastSpeech2), which served as the basis for the implementation of the depth-wise convolutions used in LightSpeech.
- [Coqui AI's excellent TTS toolkit](https://github.com/coqui-ai/TTS), which was used for the Stochastic Duration Predictor and inspired the loss weighing we do.
- [Jungil Kong's HiFi-GAN implementation](https://github.com/jik876/hifi-gan), which is used vocoding mel spectrograms produced by our TTS system.
