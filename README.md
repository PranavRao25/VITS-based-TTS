# VITS-based-TTS
EE5171 Course Project on experiments and analysis of the ESPNet VITS model for TTS task

# Team 7: ESPnet VITS Text-to-Speech Project Report

**Project:** Multispeaker text-to-speech using LibriTTS and ESPnet

**Experiments:**

- VITS conditioned on speaker IDs (SID)
- VITS conditioned on speaker embeddings (x-vectors)

| Item | Value |
|---|---|
| Hardware | 2 NVIDIA RTX 4090 GPUs, 24 GB VRAM each |
| Recipe directory | `/home/grp7/libritts_tts1` |
| ESPnet installation | `/home/Apps/espnet` |
| Dataset root | `/home/malavika/DATABASE/libritts_espnet_frmt` |
| Working sample rate | 24,000 Hz |

---

## 1. Project objective

The objective is to build a multispeaker TTS system using the ESPnet VITS architecture and LibriTTS data.

VITS combines speech generation with adversarial training. It learns to generate speech conditioned on input text and, in a multispeaker setting, speaker information.

Two conditioning strategies are planned:

1. **Speaker-ID conditioning:** Each speaker is represented by an integer ID.
2. **X-vector conditioning:** Each speaker is represented by a numerical speaker embedding.

The goal is to train the two configurations under comparable conditions and eventually compare generated speech quality, speaker identity preservation, training behavior, and computational requirements.

The first experiment is the speaker-ID version because it does not require a separate pretrained speaker-embedding extractor.

---

## 2. Environment and directory setup

### 2.1 Identify the ESPnet installation and project directory

The project is being run from:

```bash
cd /home/grp7/libritts_tts1
```

The ESPnet source installation is:

```
/home/Apps/espnet
```

The recipe uses the ESPnet `egs2` framework, specifically the TTS recipe template:

```
/home/Apps/espnet/egs2/TEMPLATE/tts1/tts.sh
```

The recipe's `tts.sh` was repaired as a symbolic link to the template script. Its link target was:

```
/home/Apps/espnet/egs2/TEMPLATE/tts1/tts.sh
```

The purpose of the script is to coordinate the end-to-end recipe stages: data formatting, speaker preparation, tokenization, statistics collection, training, inference, evaluation, and model packaging.

### 2.2 Dataset location and directory structure

The LibriTTS data is located at:

```
/home/malavika/DATABASE/libritts_espnet_frmt/
├── train/
├── dev-clean/
└── test-clean/
```

The three splits serve different purposes:

- **train:** model training.
- **dev-clean:** validation during training and model selection.
- **test-clean:** final evaluation after training.

> **Important distinction:** The prepared dataset was not copied into the project directory in the steps documented here. Instead, symbolic links were created inside the ESPnet recipe so the recipe can access the existing dataset without duplicating all the audio files.

The symbolic links are:

```
data/train      -> /home/malavika/DATABASE/libritts_espnet_frmt/train
data/dev-clean  -> /home/malavika/DATABASE/libritts_espnet_frmt/dev-clean
data/test-clean -> home/malavika/DATABASE/libritts_espnet_frmt/test-clean
```

To inspect the links:

```bash
cd /home/grp7/libritts_tts1
ls -l data/
```

The `-l` option displays symbolic-link targets.

### 2.3 Dataset manifests

Each split contains Kaldi-style metadata files: `wav.scp`, `text`, `utt2spk`, `spk2utt`, `spk2gender`.

Their functions are:

| File | Purpose |
|---|---|
| `wav.scp` | Maps each utterance ID to its audio file path. |
| `text` | Maps each utterance ID to its transcript. |
| `utt2spk` | Maps each utterance to its speaker ID. |
| `spk2utt` | Maps each speaker ID to its utterances. |
| `spk2gender` | Stores speaker gender metadata. |

These manifests are essential because ESPnet uses them to associate audio, text, and speaker identity.

The training split audit established the following:

- 149,329 training utterances.
- 2,558 distinct training speakers.
- Approximately 243.96 hours of audio.
- All checked audio paths existed.
- All checked audio was sampled at 24 kHz.

### 2.4 Why use 24 kHz?

The dataset's checked audio sample rate was 24,000 Hz. The custom training configurations were therefore adjusted to use 24 kHz rather than blindly retaining the 22,050 Hz values from the original example configuration. This keeps the declared sample rate consistent with the prepared audio.

---

## 3. Preparing the training configurations

The original configuration files were:

```
conf/tuning/train_xvector_vits.yaml
conf/tuning/decode_vits.yaml
```

Three custom configuration files were created:

```
conf/tuning/train_multispk_xvector_vits.yaml
conf/tuning/train_multispk_sid_vits.yaml
conf/tuning/decode_multispk_vits.yaml
```

The two training configurations were initially copied from the x-vector VITS example, and the custom decode configuration was copied from the existing VITS decode configuration.

### 3.1 Speaker-ID configuration

The configuration file is:

```
conf/tuning/train_multispk_sid_vits.yaml
```

The important settings include:

```yaml
spks: 2559          # 2,558 dataset speakers plus the reserved speaker ID 0 used by the recipe's SID mapping
spk_embed_dim: 0    # this configuration does not use continuous speaker embeddings
```

The model receives speaker IDs as integers. ESPnet's VITS generator supports this through a speaker embedding table.

### 3.2 X-vector configuration

The alternative configuration is:

```
conf/tuning/train_multispk_xvector_vits.yaml
```

Its relevant settings are:

```yaml
spks: -1
spk_embed_dim: 512
```

This configuration is intended for 512-dimensional speaker embeddings rather than integer speaker IDs.

### 3.3 Audio and training parameters

Both custom training configurations were adjusted to use:

```yaml
fs: 24000
sampling_rate: 24000
```

The original experiment settings included:

```yaml
num_iters_per_epoch: 1000
max_epoch: 50
batch_bins: 5000000
```

`num_iters_per_epoch` controls the configured number of training iterations per epoch, while `max_epoch` specifies the maximum number of epochs. The large initial `batch_bins` value contributed to an experiment that subsequently encountered GPU out-of-memory errors. A lower-memory configuration with `1000000` bins was later created.

---

## 4. Understanding the ESPnet recipe stages

The main script is `tts.sh`. Its stages orchestrate the training pipeline.

| Stage | Function |
|---|---|
| 1 | Prepare recipe data directories. |
| 2 | Format audio and create the raw-data dump. |
| 3 | Prepare speaker IDs or speaker embeddings. |
| 4 | Filter utterances according to length constraints. |
| 5 | Generate the text token list. |
| 6 | Collect training and validation statistics. |
| 7 | Train the VITS model. |
| 8 | Generate speech using the trained model. |
| 9 | Evaluate or score generated speech. |
| 10 | Package the model. |
| 11 | Upload the model, if configured. |

The `--stage` and `--stop_stage` arguments are inclusive. For example, `--stage 5 --stop_stage 7` attempts Stages 5, 6, and 7.

Because the dataset directories and manifests had already been prepared, the experiment began at Stage 2 rather than repeating Stage 1.

---

## 5. Speaker-ID experiment: initial data formatting

The first major run used this command:

```bash
./tts.sh \
  --stage 2 \
  --stop_stage 7 \
  --ngpu 1 \
  --lang en \
  --feats_type raw \
  --fs 24000 \
  --tts_task gan_tts \
  --use_sid true \
  --use_spk_embed false \
  --train_set train \
  --valid_set dev-clean \
  --test_sets "test-clean" \
  --train_config conf/tuning/train_multispk_sid_vits.yaml \
  --inference_config conf/tuning/decode_vits.yaml \
  --dumpdir dump_multispk_sid \
  --expdir exp/vits_multispk_sid
```

### Explanation of the command

- `--stage 2 --stop_stage 7`: run the recipe from audio formatting through training.
- `--ngpu 1`: request one GPU.
- `--lang en`: configure English text processing.
- `--feats_type raw`: use raw waveform input rather than a precomputed acoustic feature dataset.
- `--fs 24000`: specify 24 kHz audio.
- `--tts_task gan_tts`: select ESPnet's GAN-based TTS training task, which is used for VITS.
- `--use_sid true`: use speaker IDs.
- `--use_spk_embed false`: do not use x-vectors in this experiment.
- `--train_set train`: select the training split.
- `--valid_set dev-clean`: select the validation split.
- `--test_sets test-clean`: select the test split.
- `--train_config`: specify the custom speaker-ID VITS configuration.
- `--inference_config`: specify the inference configuration.
- `--dumpdir`: store formatted data and token-list artifacts.
- `--expdir`: store experiment logs, statistics, and checkpoints.

### Stage 2

ESPnet reported successful validation of the copied data directory and began audio formatting:

```
Stage 2: Format wav.scp
utils/validate_data_dir.sh: Successfully validated data-directory
scripts/audio/format_wav_scp.sh
[info]: without segments
```

This indicated that the recipe was formatting the audio and preparing the data dump. The original dataset was not modified by this stage.

---

## 6. Stage 5: resolving the text-tokenization problem

Stage 5 generates the vocabulary or token list from the training transcripts.

The initial error was:

```
RuntimeError: Please install espnet with `pip install espnet[tts]`
```

The traceback occurred during text tokenization with:

```
--token_type phn
--cleaner tacotron
--g2p g2p_en
```

These settings mean ESPnet was preparing English text for phoneme-based modeling, using the Tacotron text cleaner and the English grapheme-to-phoneme converter.

### 6.1 Diagnosing missing Python packages

The ESPnet Python environment was identified as:

```
/home/Apps/espnet/tools/miniconda/envs/espnet/bin/python3
```

The following checks were run:

```bash
/home/Apps/espnet/tools/miniconda/envs/espnet/bin/python3 -c \
  "import tacotron_cleaner; print('tacotron_cleaner OK')"

/home/Apps/espnet/tools/miniconda/envs/espnet/bin/python3 -c \
  "import jaconv; print('jaconv OK')"
```

Both imports failed with:

```
ModuleNotFoundError: No module named 'tacotron_cleaner'
ModuleNotFoundError: No module named 'jaconv'
```

An attempted installation using:

```bash
/home/Apps/espnet/tools/miniconda/envs/espnet/bin/python3 -m pip install tacotron-cleaner jaconv
```

failed because the configured package index could not find a distribution under the name `tacotron-cleaner`.

The broader package `espnet_tts_frontend` was then installed as an alternative dependency source.

### 6.2 Correcting the text input path

A second issue was that the `srctexts` argument had been supplied incorrectly. The generated command referenced a path resembling:

```
dump_multispk_sid/raw/srctexts
```

The text source was empty or incorrectly located. We found the actual text files and supplied the correct path. After correcting the text source and resolving the text-processing issue, the recipe progressed to statistics collection.

---

## 7. Stage 6: collecting statistics and fixing the pyworld dependency

After tokenization progressed, the statistics-collection step failed with:

```
ModuleNotFoundError: No module named 'pyworld'
```

The traceback showed that ESPnet imported its pitch-extraction implementation, which imports `pyworld`.

The dependency was addressed by installing it into the ESPnet Python environment:

```bash
/home/Apps/espnet/tools/miniconda/envs/espnet/bin/python3 -m pip install pyworld
```

The subsequent statistics command progressed further, indicating that the `pyworld` import issue was no longer the immediate blocker.

### 7.1 Why Stage 6 is necessary

Stage 6 collects dataset statistics used by ESPnet during training. The logged command included:

```
--collect_stats true
--feats_extract fbank
--pitch_extract_conf ...
--energy_extract_conf ...
```

Although the VITS generator ultimately works with waveform audio, the ESPnet training task can still use feature, pitch, and energy extraction components during preprocessing or statistics collection.

The command also referenced the speaker-ID and text-integer inputs:

```
dump_multispk_sid/raw/train/utt2sid,sids,text_int
dump_multispk_sid/raw/dev-clean/utt2sid,sids,text_int
```

These inputs allow the recipe to associate training utterances with speaker IDs and tokenized text.

### 7.2 Statistics collection was killed

The first statistics run was terminated:

```
bash: ... Killed
run.pl: 1 / 32 failed
```

This was consistent with a system-memory pressure problem, although the exact operating-system reason was not conclusively confirmed.

The run was initially configured to use 32 parallel jobs. Reducing the number of jobs was therefore the next troubleshooting step.

### 7.3 Rerunning Stage 6 with fewer jobs

The command was changed to use four jobs:

```bash
./tts.sh \
  --stage 6 \
  --stop_stage 6 \
  --nj 4 \
  --ngpu 1 \
  --lang en \
  --feats_type raw \
  --fs 24000 \
  --tts_task gan_tts \
  --use_sid true \
  --use_spk_embed false \
  --train_set train \
  --valid_set dev-clean \
  --test_sets test-clean \
  --train_config conf/tuning/train_multispk_sid_vits.yaml \
  --inference_config conf/tuning/decode_vits.yaml \
  --dumpdir dump_multispk_sid \
  --expdir exp/vits_multispk_sid
```

`--nj 4` reduces the number of parallel recipe jobs. Reducing `--nj` was appropriate for the statistics-collection failure because the earlier command had launched 32 jobs.

The subsequent move to Stage 7 indicates that the experiment progressed beyond the earlier statistics-collection blocker.

---

## 8. Stage 7: first training attempt and CUDA out-of-memory errors

The first training attempts failed during model training.

The log inspection command was:

```bash
grep -R -n -E 'Traceback|Error|Exception|CUDA out of memory|Killed|ERROR' \
  exp/vits_multispk_sid exp/tts_stats_raw_phn_tacotron_g2p_en 2>/dev/null | tail -40
```

It revealed explicit GPU memory failures, including:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 262.00 MiB.
GPU 0 ... 243.50 MiB is free.
```

Other failures showed GPU 1 also running out of memory.

### 8.1 Why the NCCL message appeared

The terminal output included:

```
rank 0 nranks 2
torch.multiprocessing.spawn.ProcessExitedException
```

These messages initially suggested a distributed-training issue. The detailed logs showed that the underlying failure was actually CUDA out of memory.

NCCL is the communication library used for distributed GPU training. When a worker fails, other workers can be terminated, and the outer process can report a `ProcessExitedException`.

Therefore, the first actionable issue was reducing memory consumption and ensuring the run used only one visible GPU.

### 8.2 Checking GPU status

The following command was run:

```bash
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"; nvidia-smi
```

The output showed:

```
CUDA_VISIBLE_DEVICES=not_set
```

and two RTX 4090 GPUs. At the time of this diagnostic, both GPUs were largely idle because the failed training workers had exited.

Explicitly setting GPU visibility became part of the next run.

---

## 9. Creating a lower-memory training configuration

Rather than overwrite the original configuration, a separate configuration was created:

```
conf/tuning/train_multispk_sid_vits_lowmem.yaml
```

The command was:

```bash
cp conf/tuning/train_multispk_sid_vits.yaml \
   conf/tuning/train_multispk_sid_vits_lowmem.yaml && \
sed -i 's/^batch_bins: 5000000/batch_bins: 1000000/' \
   conf/tuning/train_multispk_sid_vits_lowmem.yaml && \
grep -E 'batch_bins|num_workers|segment_size' \
   conf/tuning/train_multispk_sid_vits_lowmem.yaml
```

This copied the existing SID configuration and changed `batch_bins: 5000000` to `batch_bins: 1000000`.

The lower `batch_bins` setting reduces the amount of data ESPnet attempts to group into a batch, helping reduce GPU memory requirements. The original configuration remained available for reference.

---

## 10. Retrying Stage 7 on one visible GPU

The next training command used explicit GPU visibility and the lower-memory configuration:

```bash
CUDA_VISIBLE_DEVICES=0 ./tts.sh \
  --stage 7 \
  --stop_stage 7 \
  --ngpu 1 \
  --lang en \
  --feats_type raw \
  --fs 24000 \
  --tts_task gan_tts \
  --use_sid true \
  --use_spk_embed false \
  --train_set train \
  --valid_set dev-clean \
  --test_sets test-clean \
  --train_config conf/tuning/train_multispk_sid_vits_lowmem.yaml \
  --inference_config conf/tuning/decode_vits.yaml \
  --dumpdir dump_multispk_sid \
  --expdir exp/vits_multispk_sid
```

This command has three key differences from the earlier run:

1. `CUDA_VISIBLE_DEVICES=0` makes physical GPU 0 the only visible GPU to the process.
2. `--ngpu 1` asks ESPnet to use one GPU.
3. `train_multispk_sid_vits_lowmem.yaml` uses the lower `batch_bins` setting.

The same `dumpdir` and `expdir` were retained so the run would use the previously prepared data and experiment directory.

### Progress after the memory adjustment

A subsequent run was tried for two epochs. Two epochs are useful as a smoke test: they help establish whether the model can initialize, perform forward and backward passes, and continue training without the earlier immediate failures.

---

## 11. Training duration and early stopping

The configuration used:

```yaml
max_epoch: 50
num_iters_per_epoch: 1000
```

---

## 12. Training logs and Matplotlib curves

The main training experiment directory is:

```
/home/grp7/libritts_tts1/exp/vits_multispk_sid/
```