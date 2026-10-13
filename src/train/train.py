# SET UP LANGUAGE OF THE MODEL
# Other languages are support like Japanese, Chinese Mandarin and many others....
lang = 'English'

# PRE TRAINED MODEL
# Currently utilzing the Pre-trained VITs LJSpeech model, but can easily replace...
tag = 'kan-bayashi/ljspeech_vits'

# Replace the tag with any of the following acoustic models below
# kan-bayashi/ljspeech_tacotron2, kan-bayashi/ljspeech_fastspeech ,
# kan-bayashi/ljspeech_fastspeech2, kan-bayashi/ljspeech_conformer_fastspeech2, 
# kan-bayashi/ljspeech_vits

# VOCODER TAG
# Since VITS doesnt require a vocoder we arent utilizing one.
# That doesnt stop you from using one though and experimenting.
vocoder_tag = "none"

# Replace the tag with any of the following vocoders 
# parallel_wavegan/ljspeech_parallel_wavegan.v1, parallel_wavegan/ljspeech_full_band_melgan.v2
# parallel_wavegan/ljspeech_multi_band_melgan.v2, parallel_wavegan/ljspeech_hifigan.v1 ,
# parallel_wavegan/ljspeech_style_melgan.v1

# IMPORTS
import time
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

# MODEL INITIALIZATION
text2speech = Text2Speech.from_pretrained(
    model_tag=str_or_none(tag),
    vocoder_tag=str_or_none(vocoder_tag),
    device="cuda",
    
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

# SPEECH SYNTHESIS

# speech synthesis
input_text = 'This is what you have been waiting for!'
with torch.no_grad():
    start = time.time()
    wav = text2speech(input_text)["wav"]
rtf = (time.time() - start) / (len(wav) / text2speech.fs)
print(f"RTF = {rtf:5f}")

# let us listen to generated samples
# from IPython.display import display, Audio
# display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))

