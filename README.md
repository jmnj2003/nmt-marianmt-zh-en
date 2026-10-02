# Chinese → English Neural Machine Translation (MarianMT Fine-Tuning)

Fine-tuning the pre-trained **Helsinki-NLP MarianMT** model (`opus-mt-zh-en`) on 30,000 Mandarin–English sentence pairs from the OPUS OpenSubtitles corpus. Three hyperparameter setups are compared on loss and **BLEU**.

**Best result: 21.72 test BLEU** (learning rate 5e-5, batch size 16, 3 epochs).

<p align="center">
  <img src="assets/training_curves.png" alt="Train loss, validation loss and validation BLEU over 3 epochs for the three experiments" width="900">
</p>

---

## Pipeline

```
OPUS OpenSubtitles (en–zh_CN, 22.4M pairs)
        │  filter empty / invalid pairs → sample 30,000 pairs (seed 42)
        ▼
Split 80 / 10 / 10  →  24,000 train · 3,000 val · 3,000 test
        │  MarianTokenizer, max length 128
        ▼
Fine-tune Helsinki-NLP/opus-mt-zh-en
  AdamW (weight decay 0.01) · linear schedule with 10% warmup
  mixed precision (AMP) · gradient clipping at 1.0
        │  keep the checkpoint with the best validation BLEU
        ▼
Evaluate on the test set: beam search (4 beams) + corpus BLEU (sacreBLEU)
```

## Data

| | |
|---|---|
| Source | [OPUS OpenSubtitles v2024](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles), `en–zh_CN` Moses files |
| Full corpus | 22,394,812 aligned lines (22,373,739 valid after filtering) |
| Sample used | 30,000 random pairs (seed 42) |
| Split | 24,000 train / 3,000 validation / 3,000 test |

The notebook downloads the corpus (about 594 MB) once and caches the split as pickle files in `data/`.

## Experiments

Each run fine-tunes the model from the same pre-trained weights for 3 epochs.

| Experiment | LR | Batch | Final train loss | Final val loss | Final val BLEU | **Test loss** | **Test BLEU** |
|---|---|---|---|---|---|---|---|
| Baseline | 3e-5 | 16 | 1.718 | 1.911 | 20.34 | 1.859 | 21.44 |
| Higher LR | 5e-5 | 16 | **1.522** | **1.849** | 20.70 | **1.798** | **21.72** |
| Larger batch | 5e-5 | 32 | 1.658 | 1.878 | **21.10** | 1.847 | 21.54 |

**Findings**

- **The higher learning rate helped.** Going from 3e-5 to 5e-5 at batch size 16 gave the lowest loss everywhere and the best test BLEU.
- **Validation BLEU and test BLEU picked different winners.** The larger-batch run had the best validation BLEU, but the higher-LR run won on the test set. Validation BLEU was measured on a 50-batch subset to save time, and the gaps between runs are under 0.3 BLEU, so the three setups are close.
- **Training was still improving.** Validation loss and BLEU kept improving in epoch 3 for every run, so more epochs would probably help.
- **Larger batches cost more time.** Batch size 32 was about 4× slower per epoch on the GPU used (about 23 min vs 6 min).

## Qualitative test: a long paragraph

To test beyond short subtitle lines, each model translated a four-sentence literary paragraph about a chase scene (in the notebook). All three models kept the overall story but made word-level mistakes ("builts", "speaked", "foot" instead of "footsteps"). Two likely reasons:

- **Domain mismatch:** OpenSubtitles lines are short and conversational, while the test paragraph is long and descriptive.
- **Length:** the paragraph is close to the 128-token limit, and the model was trained on single short lines.

This gap between BLEU and real-world quality is the main motivation for the next steps below.

## Next steps

- **Tokenize targets with the target tokenizer.** MarianMT uses separate SentencePiece models for source and target. The labels here were tokenized with `tokenizer(tgt)`, which applies the *Chinese* source model to English text. Switching to `tokenizer(text_target=tgt)` should give cleaner English subwords and is the first thing to fix.
- **Measure the zero-shot baseline.** Score the pre-trained model before fine-tuning, so the gain from fine-tuning can be measured.
- Train for more epochs, and validate on the full validation set.
- Add longer, non-subtitle text (news, literature) to reduce domain mismatch.

## Getting started

```bash
git clone https://github.com/jmnj2003/nmt-marianmt-zh-en.git
cd nmt-marianmt-zh-en
pip install -r requirements.txt
jupyter notebook nmt_marianmt_zh_en.ipynb
```

- A CUDA GPU is strongly recommended. Mixed precision turns on automatically when one is available.
- The first cell downloads the 594 MB corpus and writes `data/*.pkl`. Later runs reuse that cache.
- Checkpoints are saved to `checkpoints/<experiment>/best_model.pt`. Both folders are git-ignored.

## Project structure

```
├── nmt_marianmt_zh_en.ipynb   # data prep, dataset, trainer, experiments, comparison
├── requirements.txt
└── assets/
    └── training_curves.png
```

## Tech stack

Python · PyTorch · Hugging Face Transformers (MarianMT) · sacreBLEU · pandas · Matplotlib

## Acknowledgements

- [Helsinki-NLP / OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) for the pre-trained model
- [OPUS OpenSubtitles](https://opus.nlpl.eu/) for the parallel corpus (P. Lison and J. Tiedemann, 2016)
