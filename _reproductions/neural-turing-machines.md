---
title: Neural Turing Machines
paper: Graves, Wayne & Danihelka (2014)
paper_url: https://arxiv.org/abs/1410.5401
code: https://github.com/mgupta8143/neural-turing-machines
status: in progress
date: 2026-09-22
summary: The copy task, with the LSTM baseline trained and the NTM's memory, heads and controllers implemented.
---

The Neural Turing Machine pairs a neural network controller with an external memory matrix it
learns to read from and write to. The paper's claim is that this lets it learn an *algorithm*
rather than memorise a mapping: trained to copy sequences of up to 20 vectors, it keeps copying
at lengths it has never seen, while an LSTM of 80 times the size falls apart past its training
range.

This is a reproduction of the copy task, with three models: the LSTM baseline, the NTM with a
feed-forward controller, and the NTM with an LSTM controller.

## The task

The network reads a sequence of random 8-bit vectors, then a delimiter flag, and must output the
same vectors back while its input is blank. The sequence length is random between 1 and 20, so
each example has `2L + 1` timesteps: L vectors, one delimiter, then L blank steps where the
recall happens. Only those last L steps are scored.

## Status

| Piece | State |
|---|---|
| Copy task and training loop | done |
| LSTM baseline (3 × 256) | trained, 1M sequences |
| NTM memory, addressing, heads, both controllers | done, tested |
| NTM copy-task runs | running |
| Memory-use figure (paper Figure 6) | not started |

## LSTM baseline

One run of 1M sequences on a Colab T4 at batch size 16, learning rate 1e-4. Every other setting
matches the paper: RMSProp with momentum 0.9, gradients clipped elementwise to (−10, 10),
cross-entropy reported in bits per sequence.

| Sequences seen | This run | Paper (read off Figure 3) |
|---|---|---|
| 10k | 83.0 bits | above 10 |
| 100k | 17.7 bits | ~5 bits |
| 200k | 8.2 bits | ~2 bits |
| 500k | 2.3 bits | ~0.5 bits |
| 1M | 0.9 bits | ~0.5 bits |

**What matched.** The shape of the curve, a fast drop from random guessing at about 84 bits then
a long slow tail. Length 10 is copied perfectly and length 20 almost perfectly. At lengths 30, 50
and 120 the model copies roughly the first dozen vectors and then hedges every bit at about 0.5,
which is the paper's headline LSTM result: no general copying procedure, just as much as it can
hold.

**What didn't.** Learning was 2–3× slower per sequence than the paper's. The main cause is batch
size: at 16 sequences per update, 1M sequences is only about 62k weight updates, where the paper's
(unstated, but most likely 1) would give 1M. Smaller contributors: PyTorch's standard RMSProp
rather than Graves' 2013 variant, a zeroed rather than learned starting state, and about 2% fewer
parameters (1,328,136 against the paper's 1,352,969).

## NTM implementation

The model is split so the pieces swap: memory operations, the addressing pipeline, heads, and
controllers are all independent.

```
src/models/ntm/memory.py       read, write, and the four addressing stages of Figure 2
src/models/ntm/heads.py        ReadHead and WriteHead: one Linear -> weighting -> read or write
src/models/ntm/controllers.py  FeedForwardController and LSTMController, same interface
src/models/ntm/ntm.py          controller + heads + memory + output layer, one timestep at a time
```

Parameter counts land close to the paper's: 16,096 against 17,162 for the feed-forward
controller, and 62,660 against 67,561 for the LSTM controller.

### Two things worth recording

**Symmetry has to be broken at initialisation.** With every memory row identical and every head
weighting uniform, all 128 locations are interchangeable, so their gradients are identical and
they never differentiate: the memory behaves like a single slot and training plateaus near
chance. Starting the learned memory and weightings from small random values instead fixes it. On
a fixed-length-5 copy, 1,200 updates take the cost from 40 bits to 1.1 with random
initialisation, against 34 with constant initialisation.

**Sharpening needs the log form.** Equation 9 raises the weighting to a power and renormalises.
Computed literally, small weights raised to a high power underflow float32 to zero, and the
renormalised weighting comes back as all zeros, leaving the head attending to nothing. Using
`softmax(γ · log w)`, which is the same expression algebraically, removes both that and an
epsilon-sized disagreement with the paper's formula.

Each addressing stage is checked against hand-worked expectations, and the whole pipeline against
a loop-by-loop transcription of equations 5–9 written straight from the paper.

## Next

Copy-task runs for both NTM variants, then the generalisation figure at lengths 10 to 120 beside
the LSTM's, and the memory-use heatmaps that show the write head walking along memory and the read
head retracing it.
