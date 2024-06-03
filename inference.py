import torch
import torchaudio
from glob import glob
import numpy as np
import os
import torchaudio.functional as F
from vocos import Vocos

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model ckpt")
    parser.add_argument("--config_path", required=True, help="Path to model config (.yaml)")
    parser.add_argument("--output_path", required=True, help="Path to write WAV file")
    parser.add_argument("--mel_input", required=False, type=str, help="mel input")
    parser.add_argument("--audio_input", required=False, type=str, help="audio input")
    args = parser.parse_args()

    checkpoint_path = args.model
    config_path = args.config_path
    audio_path = args.audio_input
    mel_path = args.mel_input

    ## load model for inference
    model = Vocos.from_hparams(config_path)
    raw_model = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(raw_model['state_dict'], strict=False)
    model.eval()

    # read soruce audio
    if audio_path:
        src_audio, fs = torchaudio.load(audio_path)
        if fs != 22050:
            src_audio = F.resample(src_audio, orig_freq=fs, new_freq=22050)

        # inference
        audio = model(src_audio)
    # read mel spectrogram
    elif mel_path:
        mel = torch.tensor(np.load(mel_path))
        audio = model.decode(mel)

    wav_file = f'{os.path.basename(checkpoint_path)}_{os.path.basename(audio_path)}_mod.wav'
    torchaudio.save(wav_file, audio.cpu(), 22050, )

if __name__=="__main__":
    main()