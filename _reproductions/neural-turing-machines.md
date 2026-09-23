---
title: Neural Turing Machines
paper: Graves, Wayne & Danihelka, 2014
paper_url: https://arxiv.org/abs/1410.5401
code: https://github.com/mgupta8143/neural-turing-machines
status: in progress
date: 2026-09-22
summary: The copy task. LSTM baseline trained to 0.9 bits; NTM implemented and running.
---

An NTM is a controller network attached to a memory matrix it learns to read and write with
differentiable attention. The paper's claim is that this makes it learn a procedure rather than
memorise a mapping: trained on sequences up to length 20, it keeps copying at lengths it never
saw, while an LSTM does not.

I am reproducing the copy task with three models: the LSTM baseline, an NTM with a feed-forward
controller, and an NTM with an LSTM controller.

## Task

Input is a sequence of L random 8-bit vectors, then a delimiter on a 9th channel, then L blank
steps. The target is the same L vectors, scored only on the blank steps. L is uniform on 1 to 20.

## Status

| | |
|---|---|
| Copy task, training loop, figures | done |
| LSTM baseline, 1M sequences | done |
| NTM memory, addressing, heads, both controllers | done, tested |
| NTM copy-task runs | running |
| Memory-use figure (paper Fig. 6) | not started |

## LSTM baseline

3 layers of 256 units plus a linear readout, 1,328,136 parameters against the paper's 1,352,969.
RMSProp, momentum 0.9, gradients clipped to (-10, 10), 1M sequences on a Colab T4. I used batch
size 16 and learning rate 1e-4; the paper uses 3e-5 and does not state a batch size.

| Sequences | Mine | Paper, read off Fig. 3 |
|---|---|---|
| 10k | 83.0 bits | above 10 |
| 100k | 17.7 | ~5 |
| 200k | 8.2 | ~2 |
| 500k | 2.3 | ~0.5 |
| 1M | 0.9 | ~0.5 |

Matched: the curve shape, near-perfect copying at length 10 and 20, and failure past 20, where it
reproduces about the first dozen vectors and then outputs ~0.5 for every bit.

Did not match: learning was 2-3x slower per sequence. Batch size is the likely cause. At 16
sequences per update, 1M sequences is 62k updates; at batch size 1 it would be 1M. Minor
differences: PyTorch's RMSProp rather than the Graves 2013 variant, zeroed rather than learned
initial state, 2% fewer parameters.

## NTM implementation

```
memory.py       read, write, and the four addressing stages of Fig. 2
heads.py        ReadHead and WriteHead: one Linear -> weighting -> read or write
controllers.py  FeedForwardController and LSTMController, same interface
ntm.py          controller + heads + memory + output layer, one timestep at a time
```

16,096 parameters with the feed-forward controller against the paper's 17,162, and 62,660 with
the LSTM controller against 67,561.

Two things that cost me time:

**Initialisation has to break symmetry.** With identical memory rows and uniform head weightings,
all 128 locations are interchangeable, their gradients are identical, and they never
differentiate. The memory then behaves like one slot and training plateaus near chance. Starting
the learned memory and weightings from small random values fixes it: on a fixed-length-5 copy,
1,200 updates reach 1.1 bits with random initialisation and 34 bits with constant initialisation.

**Sharpening underflows.** Equation 9 raises the weighting to a power and renormalises. Done
literally, small weights at a high power underflow float32 to zero, so the weighting becomes all
zeros and the head attends to nothing. `softmax(gamma * log w)` is the same expression and is
stable.

I checked each addressing stage against hand-worked values, and the whole pipeline against a
loop-by-loop transcription of equations 5 to 9.

## Next

Runs for both NTM variants, the generalisation figure at lengths 10 to 120 next to the LSTM's,
and the memory-use heatmaps.
