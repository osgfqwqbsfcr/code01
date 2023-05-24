import argparse
import json
import os

import torch
import numpy as np

import utils
from models import (
    SynthesizerTrn
)
from text import text_to_sequence
from text.symbols import symbols
from scipy.io.wavfile import write

torch.backends.cudnn.benchmark = False
global_step = 0
device = None
MAX_WAV_VALUE = 32768.0


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output audio path')
    parser.add_argument('-cp', '--checkpoint_path', type=str, required=True, help='Checkpoint path')
    parser.add_argument('-t', '--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('-cb', '--codebook_path', type=str, required=True, help='Path of a codebook')

    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = utils.HParams(**config)
    hparams.checkpoint_path = args.checkpoint_path
    return hparams, args


def main():
    assert torch.cuda.is_available(), "CPU inference is not supported."

    global device
    device = torch.device('cuda')

    hps, a = get_hparams()
    inference(hps, a)


def inference(hps, a):
    net_g = SynthesizerTrn(len(symbols), 80, **hps.model).cuda()

    # if hasattr(hps.model, 'ri_upsample_rates'):
    #     net_g.dec.reinitialize_networks(hps.model.inter_channels, hps.model.ri_upsample_rates, hps.model.ri_upsample_initial_channel,
    #                                            hps.model.ri_upsample_kernel_sizes, hps.model.generator_freeze_blocks)
    #     net_g.cuda()

    _ = utils.load_checkpoint(hps.checkpoint_path, net_g, None)

    net_g.eval()
    net_g.remove_weight_norm()

    with torch.no_grad():
        x, sf = get_input_data(a.text, a.codebook_path)
        x, sf = x.unsqueeze(0).cuda(), sf.unsqueeze(0).cuda()
        x_lengths = torch.LongTensor([x.size(1)]).cuda()

        y_hat, attn, mask, *_ = net_g(x, x_lengths, sf)
        # output_file = os.path.join(hps.output_dir, 'synthesized_speech.wav')
        output_file = a.output_file
        audio = y_hat * MAX_WAV_VALUE
        write(output_file, 22050, audio.cpu().numpy().astype('int16'))
        print(output_file)


def get_input_data(text, codebook_path):
    text = get_text(text)
    sf = np.load(codebook_path)
    sf = torch.FloatTensor(sf.astype(np.float32))
    return text, sf


def get_text(text):
    text_norm = text_to_sequence(text)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


if __name__ == "__main__":
    main()
