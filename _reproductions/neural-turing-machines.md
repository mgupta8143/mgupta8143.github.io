---
title: Neural Turing Machines
paper: Graves, Wayne & Danihelka, 2014
venue: NTM
authors: Alex Graves, Greg Wayne, Ivo Danihelka
published_in: arXiv:1410.5401, 2014
paper_url: https://arxiv.org/abs/1410.5401
code: https://github.com/mgupta8143/neural-turing-machines
status: done
date: 2026-09-26
summary: Copy and associative recall reproduced from scratch. Both NTMs reach zero cost, the LSTM baseline never does, and shrinking the memory below the input size shows the network packing two vectors into one slot.
---

I chose a reproduction because you cannot fudge one. The answer is already published: either your
curve lands on theirs or it does not, and if it does not you have to say so and work out why.

It also covers most of what research engineering actually is. You have to read the method closely
enough to implement it, and build enough infrastructure to run it. Then you have to publish the
gap.

So I rebuilt the Neural Turing Machine from the paper and ran two of its five tasks: copy and
associative recall. Both NTM variants reach exactly zero cost on both tasks and the LSTM baseline
never does. That is the paper's central claim, and it held up. On associative recall the
reproduction generalises *better* than the published numbers at every point I measured, and I
cannot account for it. Four details the paper omits decide whether the model trains at all, and a
fifth thing it never mentions decides how fast.

Then, once I had something that worked, I ran an experiment the paper does not: shrinking the
memory below the size of the input. It packs two vectors into one memory slot and recovers both.

## The claim worth checking

The paper argues that a neural network given an external memory and a differentiable way to
address it will learn *algorithms* rather than just fit the task. It supports this with Figure 6,
which shows the read and write heads tracing a clean diagonal across memory, and it goes as far
as writing out the pseudocode it believes the network discovered:

```
initialise: move head to start location
while input delimiter not seen do
  receive input vector
  write input to head location
  increment head location by 1
end while
return head to start location
while true do
  read output vector from head location
  emit output
  increment head location by 1
end while
```

That is a strong claim, and it is why I wanted to build the thing. The learning curves I believed
on sight. Whether what ends up in the memory deserves to be called an algorithm is what I wanted
to see for myself.

## Setup

**Copy** (Section 4.1): the network reads L random 8-bit vectors and a delimiter, then has to
emit the same vectors again with no input. L is drawn from 1 to 20, so an example is 2L+1
timesteps and only the last L are scored. Chance is 84 bits per sequence.

**Associative recall** (Section 4.3): an item is three six-bit vectors bounded by delimiters. An
episode shows two to six items, then a query delimiter and one of the items again. The network
must emit the item that *followed* the query. Only those three steps are scored, so chance is
18 bits.

| | Paper | Mine |
|---|---|---|
| Optimiser | RMSProp, Graves (2013) form, momentum 0.9 | same |
| Learning rate | 1e-4 NTM, 3e-5 LSTM (copy); 1e-4 all (recall) | same |
| Batch size | not stated | 1 |
| Memory | 128 × 20 | same |
| Gradient clipping | elementwise to (−10, 10) | **norm, not elementwise** |
| Training length | 500k+ sequences | 500k |

The clipping row is a real deviation and I explain it below.

Parameter counts do not reconcile, and this is not a small discrepancy:

| | copy | paper | recall | paper |
|---|---|---|---|---|
| LSTM baseline | 1,328,136 | 1,352,969 | 1,326,598 | 1,344,518 |
| NTM, feed-forward | 16,096 | 17,162 | 123,046 | 146,845 |
| NTM, LSTM controller | 65,696 | 67,561 | 65,054 | 70,330 |

Mine come out consistently smaller, and no reading of the head counts closes the gap — the next
best interpretation is 36% off rather than 16%. I think the published numbers are not
reconstructible from the published architectures. Table 2 supports that: it gives copy and
associative recall *identical* NTM settings (1 head, 100 units, 128 × 20) yet lists associative
recall as 2,769 parameters larger, although it has fewer input and output channels and must
therefore be smaller.

## What I built

```
src/models/ntm/memory.py       read, write, and the four addressing stages of Figure 2
src/models/ntm/heads.py        one Linear per head -> weighting -> read or write
src/models/ntm/controllers.py  feed-forward and LSTM, same interface
src/models/ntm/ntm.py          controller + heads + memory, one timestep at a time
src/models/lstm/lstm.py        the baseline
src/train.py                   training loop, CUDA-graph capture, logging
src/probe.py                   diagnostics logged beside the cost
src/tasks/<task>/              data.py makes the task, plots.py draws its figures
```

Everything above `src/tasks/` is task-agnostic: a new task is a data module and a settings entry.

Two of the choices here were mine, not the paper's. The NTMs are clipped by gradient **norm**:
their norms spike to hundreds of times the median, and elementwise clipping lets one of those
spikes through as an update that wipes out the addressing the model has learned. The initial
memory is **random**. With identical rows, every location gets an identical gradient and the 128
of them never differentiate.

## Copy

<figure>
  <img src="/assets/ntm-copy-curves.png" alt="Learning curves for three models on the copy task">
  <figcaption>Cost per sequence in bits. Compare with Figure 3 of the paper.</figcaption>
</figure>

| Sequences | LSTM | NTM, feed-forward | NTM, LSTM controller |
|---|---|---|---|
| 10k | 77.5 bits | **0.00** | **0.00** |
| 100k | 10.1 | 0.00 | 0.00 |
| 500k | **1.03** | **0.0000** | **0.0000** |
| first below 0.05 bits | never | 5,000 | 10,000 |

Both NTMs solve the task inside the first few thousand sequences and hold zero for the remaining
490,000. The LSTM needs the whole run to reach about one bit.

<figure>
  <img src="/assets/ntm-copy-generalisation.png" alt="NTM outputs and targets at lengths 10, 20, 30, 50 and 120">
  <figcaption>Trained on lengths up to 20. Lengths 10, 20, 30, 50 across the top; 120 below.</figcaption>
</figure>

| Test length | 10 | 20 | 30 | 50 | 80 | 120 |
|---|---|---|---|---|---|---|
| LSTM | 0.0% | 3.1% | 22.9% | 40.0% | 46.4% | 48.0% |
| NTM, feed-forward | 0.0% | 0.0% | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| NTM, LSTM controller | 0.0% | 0.0% | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

Both NTMs copy perfectly at six times their training length. This is slightly better than the
paper's own result: its Figure 4 caption reports "a few more local errors and one global error"
at length 120, where a vector is duplicated and everything after it shifts by one. Mine makes
neither mistake.

<figure>
  <img src="/assets/ntm-copy-memory.png" alt="Write and read weightings over time on a length-40 copy">
  <figcaption>Figure 6 reproduced at length 40, double the training range. Left: input, vectors
  written, write weightings. Right: output, vectors read, read weightings.</figcaption>
</figure>

Measured on the trained model, the write head advances +1.00 memory locations per timestep with
its weighting pinned at 0.999, and the read head retraces the same path. That is the pseudocode
above.

## Associative recall

<figure>
  <img src="/assets/ntm-recall-curves.png" alt="Learning curves for three models on associative recall">
  <figcaption>Compare with Figure 10. Chance is 18 bits.</figcaption>
</figure>

| | first at zero cost | cost at 500k |
|---|---|---|
| NTM, feed-forward | **37,000** | 0.0000 |
| NTM, LSTM controller | **40,000** | 0.0000 |
| LSTM | never | 6.59 |

Both NTMs fall off a cliff between 20k and 40k sequences and hold exactly zero. The LSTM grinds
from 18 bits to 6.6 over the full 500,000 and flattens out around 300,000 without solving it.
The paper puts its NTM at "near zero cost within approximately 30,000 episodes, whereas LSTM
does not reach zero cost after a million"; mine converges at 37,000.

<figure>
  <img src="/assets/ntm-recall-generalisation.png" alt="Cost against number of items per sequence for three models">
  <figcaption>Figure 11 reproduced. Trained on 2 to 6 items, tested to 20.</figcaption>
</figure>

| items per sequence | 6 | 10 | 15 | 20 |
|---|---|---|---|---|
| LSTM | 14.93 | 18.63 | 18.91 | 20.48 |
| NTM, feed-forward | **0.00** | **0.00** | 1.77 | 2.03 |
| NTM, LSTM controller | **0.00** | **0.00** | **0.00** | **0.94** |
| *paper, feed-forward* | *~0.05* | *0.1* | *1.3* | *7.8* |
| *paper, LSTM controller* | *~0.1* | *1.7* | *4.5* | *6.8* |

**Both of my NTMs beat both of the paper's at every point on this axis.** At twenty items — more
than three times the training maximum — the paper's best model is at 6.8 bits and mine is at
0.94. I have no explanation. My models are 16% and 7% *smaller* than the paper's by parameter
count, so it is not capacity, and the settings are the ones the tables specify.

<figure>
  <img src="/assets/ntm-recall-memory.png" alt="Memory use during an associative recall episode">
  <figcaption>Figure 12 reproduced. Green marks the query item, red the target and the answer,
  black the write the query is later looked up by.</figcaption>
</figure>

## Taking the memory away

By this point I had a working model and all the probing and plotting code I had written earlier
to find out why it wasn't working. New questions were suddenly cheap. The one I wanted was about
memory size.

Every published NTM result gives the network far more memory than it needs — 128 slots to hold at
most 20 vectors. Six times the room the obvious strategy requires, so the model is never under
pressure. I could not find a published run with less.

The paper knows memory size is the binding constraint and says so, in a footnote explaining why
copy generalisation eventually breaks: *"The limiting factor was the size of the memory (128
locations), after which the cyclical shifts wrapped around and previous writes were
overwritten."* But it only ever hits that wall from one side, by making sequences longer. Holding
the task fixed and taking memory away instead is easier to control and easier to measure.

Nine feed-forward NTMs, copy task, lengths 1 to 20 as before, 30,000 sequences each. The only
thing that changes between runs is the shape of the memory. The feed-forward controller is the
only honest choice here: it has no state between timesteps, so anything surviving the delimiter
had to pass through memory.

<figure>
  <img src="/assets/ntm-pressure-curves.png" alt="Bit error against sequence length for six memory sizes, and for four memories of equal total size">
  <figcaption>Left: shrinking memory, with the dotted line marking where each memory runs out of
  slots. Right: the control, where every memory holds exactly 400 numbers in a different shape.</figcaption>
</figure>

**Twelve slots is enough for twenty vectors.** Zero bit error at every length tested, which means
eight of the twenty vectors have nowhere of their own to live. Only at eight slots does error
appear in earnest.

<figure>
  <img src="/assets/ntm-pressure-decodability.png" alt="Grids of memory slot against input position, shaded by linear decodability">
  <figcaption>For every slot and every input vector: can that vector be recovered from that slot
  by a linear probe, over 400 sequences? Bright means yes.</figcaption>
</figure>

A heatmap of where the head wrote cannot tell packing from overwriting, so the test is a linear
probe fitted from each slot's contents to each input vector.

| Slots | Vectors per slot | Slots per vector | Positions recoverable |
|---|---|---|---|
| 20 | 1.00 | 1.00 | 20 / 20 |
| 12 | **1.67** | 1.00 | **20 / 20** |
| 8 | 1.25 | 1.00 | 10 / 20 |

At twenty slots there is one bright diagonal: one vector per slot, as the paper describes. At
twelve there are **two** diagonals — slot 1 holds vector 1 *and* vector 13, and both come back
out. 1.67 is exactly 20/12, so every vector is accounted for and the load is even. Slots per
vector stays at 1.00 throughout, which rules out the other explanation: nothing is being smeared
across slots.

Eight slots is where it breaks, and it breaks the wrong way round. Under more pressure it packs
*less*: 1.25 vectors per slot against 1.67 at twelve. Ten of the twenty positions are not
recoverable from memory at all. So it gives up on half the sequence and keeps the other half
clean. I had expected the error to spread out.

<figure>
  <img src="/assets/ntm-pressure-memory.png" alt="Write and read weightings for memories of 20, 12 and 8 slots">
  <figcaption>At 20 slots the write head walks one diagonal and stops. At 12 it reaches the end
  of memory, wraps to the start, and lays a second diagonal over the first.</figcaption>
</figure>

### It runs out of addresses, not room

Every memory in the control arm holds the same 400 numbers, arranged differently:

| Memory | Numbers | Parameters | Bit error at length 20 |
|---|---|---|---|
| 20 × 20 | 400 | 13,720 | **0.0%** |
| 8 × 50 | 400 | 29,086 | 24.2% |
| 4 × 100 | 400 | 54,728 | 34.0% |
| 2 × 200 | 400 | 106,024 | 35.1% |

Same storage, and error climbs from nothing to a third of the bits purely by giving the network
fewer places to address. **Twelve slots of width 20 hold 240 numbers and score 0.0%. Eight slots
of width 50 hold 400 numbers, cost twice the parameters, and score 24.2%.** More room, worse
result.

One caveat on this section: one training run per configuration. The clearest sign that matters is
16 slots scoring 7.4% where 12 scores 0.0%, which is almost certainly training variance rather
than a real non-monotonicity, and it means the exact breaking point is not pinned down.

## What matched

The shape and separation of the learning curves on both tasks. Zero cost for both NTMs and a
baseline that never gets there. Perfect copy generalisation far past the training range, and the
LSTM's failure mode including the shrinking accurate prefix. The learned copy algorithm visible
in the memory traces, at +1.00 locations per timestep. Associative
recall converged at 37,000 sequences, against the paper's approximately 30,000.

## What didn't

**The controller ordering.** The paper reports the feed-forward controller converging faster than
the LSTM controller on both tasks. On copy mine is the reverse (10k against 5k); on associative
recall they are within 3,000 sequences of each other, which is inside the seed variance I
measured. Two tasks, neither confirming the ordering.

**Parameter counts**, by 2 to 16%, as above. I believe this one is the paper's.

**Generalisation on associative recall**, where I come out better than the published result.
Still unexplained.

## What the paper leaves out

Roughly in the order of how much time each one cost me.

**RMSProp as Graves (2013) defines it.** Section 4.6 cites that form without stating it: centered,
decay 0.95, damping 1e-4 inside the square root. PyTorch's defaults are 0.99 and 1e-8, and that
1e-8 inflates the update wherever gradient variance is small, which is most of an LSTM
controller's recurrent matrix. With the defaults my LSTM-controller NTM reached zero and then
*drifted back off it*: by the end of a million sequences it was 46% wrong at length 50. With the
paper's form it sits at 0.0000 for 450,000 consecutive sequences.

**Clip the norm, not each component.** Described above. Clipped elementwise, the run diverged
partway through. Clipped by norm, it reached zero.

**The starting memory has to break symmetry.** With identical rows every location is
interchangeable, gradients are identical, and the locations never differentiate. On a
fixed-length-5 copy, 1,200 updates reach 1.1 bits with random initialisation against 34 bits with
constant.

**Sharpening underflows if written literally.** Equation 9 raises the weighting to a power. Small
weights at a high power underflow float32 to zero, the renormalisation divides by zero, and the
head attends to nothing. `softmax(γ log w)` is the same expression and is stable.

And one that decides how *fast* it trains, which is the one thing here I did not expect at all. **The range of training lengths matters more than the lengths themselves.** Holding the
model, the loop and the seed fixed and changing only the range on the copy task:

| lengths trained on | cost at 10,000 sequences | converged |
|---|---|---|
| 1 to 20 | 0.01 bits | yes, by 7,500 |
| 1 to 15 | 0.69 | yes, by 10,000 |
| **1 to 10** | **27.0, 29.8, 30.5** (three seeds) | **no** |
| fixed length 10 | 61.5 | no |

Narrowing the range does not make the task easier. At 1 to 10 it stops training at all, on three
seeds. A fixed length is worse again. My reading is that the short examples in a wide range are
what break the addressing symmetry cheaply, and everything else bootstraps off them. It is why I
ran the memory-pressure study on a range of lengths rather than pinning the sequence. Pinning it
would have measured a different failure.

## Open questions

**Does it actually read the memory, or has it memorised?** Everything the paper offers is
correlational: weightings that look like an algorithm, outputs that come out right. A model that
kept the sequence in its controller state and drove the heads along a diagonal anyway would
produce an identical Figure 6. The test is an intervention — overwrite a memory location
mid-recall and see whether the output follows. If it reads, the output changes to what you
injected. Cheap to run, since it needs no training.

**Why is my associative recall generalisation better than the paper's?** Smaller models, same
settings, better numbers at every point. I do not have a story for this.

**Does the packing code generalise?** A model trained at twelve slots has learned to put two
vectors in one place. Give it a sequence longer than it ever saw and find out whether it packs
three.

**What makes two writes to one slot survive each other?** The mechanism is described but not
explained. Either the erase vector learns to spare what is already there, or the two add vectors
land in parts of the slot that do not overlap. Both would show up in traces I already have. No new
training.

## Running it

```sh
git clone https://github.com/mgupta8143/neural-turing-machines
cd neural-turing-machines
uv sync
uv run main.py train --model ntm-ff --sequences 20000
```

It starts at 84 bits, which is chance, and should be near zero inside ten thousand sequences.
That takes a few minutes on a laptop. Everything takes `--task`, which defaults to copy.

Most of the results here came off rented A10Gs through Modal. The NTM is slower on a GPU than a
CPU if you run it naively, because it launches a few hundred tiny kernels per sequence. I capture
the training step as a CUDA graph, one per sequence length, and then the GPU is worth paying
for.
