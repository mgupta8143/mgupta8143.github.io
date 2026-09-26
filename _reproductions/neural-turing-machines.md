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

The paper is over a decade old, and that is most of why I wanted to build it. It sits just before
attention took over, and the questions it is asking — how does a network decide where to look,
how does it put something down and find it again later — are the ones attention went on to answer
differently. Working through the version that came first is a way to understand what the later
answer was actually replacing.

I picked copy and associative recall out of the five tasks because they are the simplest, which
makes them the easiest to check: both have published learning curves I could hold mine against,
and both fail in ways you can see rather than only measure. Everything was trained on rented A10G
GPUs through [Modal](https://modal.com) and on my own laptop, and all of it is in the
[code](https://github.com/mgupta8143/neural-turing-machines).

I rebuilt the Neural Turing Machine from the paper and ran two of its five tasks: copy and
associative recall. Both NTM variants reach exactly zero cost on both tasks and the LSTM baseline
never does. That is the paper's central claim, and it held up. On associative recall the
reproduction generalises *better* than the published numbers at every point I measured, and I
cannot account for it. Four details the paper omits decide whether the model trains at all, and a
fifth thing it never mentions decides how fast.

Then, once I had something that worked, I ran an experiment the paper does not: shrinking the
memory below the size of the input. It packs two vectors into one memory slot and recovers both.

I chose a reproduction because you cannot fudge one. The answer is already published: either your
curve lands on theirs or it does not, and if it does not you have to say so and work out why.

## The claim: it learns an algorithm, not a fit

The paper argues that a neural network given an external memory and a differentiable way to
address it will learn *algorithms* rather than just fit the task. It supports this with Figure 6,
which shows the read and write heads tracing a clean diagonal across memory, and it goes as far
as describing the algorithm it believes the network discovered: write each input vector to a
successive memory location, then return to the start and read them back in order.

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

| | Copy: mine | Copy: paper | Recall: mine | Recall: paper |
|---|---|---|---|---|
| LSTM baseline | 1,328,136 | 1,352,969 | 1,326,598 | 1,344,518 |
| NTM, feed-forward | 16,096 | 17,162 | 123,046 | 146,845 |
| NTM, LSTM controller | 65,696 | 67,561 | 65,054 | 70,330 |

Mine come out consistently smaller, and no reading of the head counts closes the gap. Reading
"4 heads" as four in total rather than four of each puts associative recall 56% under instead of
16%, and reading it as four read heads and one write head puts it 51% under. I think the published numbers are not
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

Two of the choices in there were mine, not the paper's — how the gradients are clipped and how
the memory starts out. Both are in the next section, which is where the real work went.

## Four omissions that decide whether it trains

Roughly in the order of how much time each one cost me.

**RMSProp as Graves (2013) defines it.** Section 4.6 cites that form without stating it: centered,
decay 0.95, damping 1e-4 inside the square root. PyTorch's defaults are 0.99 and 1e-8, and that
1e-8 inflates the update wherever gradient variance is small, which is most of an LSTM
controller's recurrent matrix. With PyTorch's defaults the LSTM-controller NTM would reach zero and then *drift back off it*
later in the run. With the paper's form the same model sits at exactly 0.0000 for its last
399,000 sequences unbroken. I no longer have the logs from the default-optimiser runs, so take
the first half of that as a recollection rather than a measurement; the second half is in
`results/copy/ntm-lstm/log.csv`.

**Clip the norm, not each component.** This is the one deviation rather than an omission — the
paper does specify elementwise clipping. The reason is measurable: with the paper's optimiser,
pre-clip gradient norms on the feed-forward NTM peak at 566 to 176,000 times their running
median over 8,000 updates. Clipping each component to (-10, 10) lets a spike like that through as
an update large enough to destroy the addressing. I switched to clipping the norm early and did
not keep a controlled comparison, so the mechanism is measured and the consequence is inferred.

**The starting memory has to break symmetry.** With identical rows every location is
interchangeable, every location gets the same gradient, and the 128 of them never differentiate —
the model sits near chance. Random starting values fix it. I found this early, before I was
keeping logs properly, so I can point at the reasoning and the code but not at a saved run.

**Sharpening underflows if written literally.** Equation 9 raises the weighting to a power. Small
weights at a high power underflow float32 to zero, the renormalisation divides by zero, and the
head attends to nothing. `softmax(γ log w)` is the same expression and is stable.

And one that decides how *fast* it trains, which is the one thing here I did not expect at all. **The range of training lengths matters more than the lengths themselves.** Holding the
model, the loop and the seed fixed and changing only the range on the copy task:

Chance differs with the range, so the comparable number is cost as a fraction of it:

| lengths trained on | cost at 10,000 sequences, as % of chance | solved by 10,000 |
|---|---|---|
| 1 to 20 | **0.0%, 3.1%, 46.8%** (three seeds) | 1 of 3 |
| 1 to 15 | 1.1% (one seed) | yes |
| **1 to 10** | **61.4%, 67.7%, 69.4%** (three seeds) | **0 of 3** |
| fixed length 10 | 76.9% (one seed) | no |

Narrowing the range does not make the task easier. The two arms do not overlap: the worst 1-to-20
seed is still better than the best 1-to-10 seed, and no 1-to-10 seed gets below 61% of chance
inside the budget. But only one of the three 1-to-20 seeds actually solved it in 10,000
sequences, so "converges in 7,500" describes a lucky seed and not the arm. The separation is
real; the speed is not as clean as one run makes it look.

My reading is that the short examples in a wide range are what break the addressing symmetry
cheaply, and everything else bootstraps off them. It is why I ran the memory-pressure study on a
range of lengths rather than pinning the sequence. Pinning it would have measured a different
failure.

## Copy: both NTMs reach zero, the LSTM never does

<figure>
  <img src="/assets/ntm-copy-curves.png" alt="Learning curves for three models on the copy task">
  <figcaption>Both NTMs reach zero cost inside the first 10,000 sequences and hold it for the
  remaining 490,000; the LSTM is still at about one bit after half a million. Compare the paper's
  Figure 3.</figcaption>
</figure>

| Sequences | LSTM | NTM, feed-forward | NTM, LSTM controller |
|---|---|---|---|
| 10k | 77.5 bits | **0.00** | **0.00** |
| 100k | 10.1 | 0.00 | 0.00 |
| 500k | **1.03** | **0.0000** | **0.0000** |
| first below 0.05 bits | never | 5,000 | 10,000 |

Both NTMs solve the task inside the first few thousand sequences. Neither holds zero perfectly
afterwards — both spike occasionally, the LSTM-controller one as late as 46,000 — but the
LSTM-controller run then sits at exactly 0.0000 for its last 399,000 sequences unbroken. The
LSTM baseline needs the whole run to reach about one bit.

<figure>
  <img src="/assets/ntm-copy-generalisation.png" alt="NTM outputs and targets at lengths 10, 20, 30, 50 and 120">
  <figcaption>Trained on lengths up to 20, the output is indistinguishable from the target at
  120 — no duplicated vector and no global shift, the two errors the paper's own Figure 4 caption
  reports at that length. Lengths 10, 20, 30 and 50 across the top; 120 below.</figcaption>
</figure>

Percentage of output bits wrong, 20 sequences per length. The LSTM row moves a point or two
between draws; the NTM zeros do not.

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
  <figcaption>The write head lays one sharp diagonal as the input arrives and the read head
  retraces the same path on the way out: +1.00 locations per timestep, weighting pinned at 0.999.
  Length 40, double the training range; the paper's Figure 6 for comparison.</figcaption>
</figure>

Measured on the trained model, the write head advances +1.00 memory locations per timestep with
its weighting pinned at 0.999, and the read head retraces the same path. That is the pseudocode
above.

## Associative recall: zero cost at 37,000 sequences

<figure>
  <img src="/assets/ntm-recall-curves.png" alt="Learning curves for three models on associative recall">
  <figcaption>Both NTMs fall off a cliff between 20k and 40k sequences and hold exactly zero.
  The LSTM grinds from 18 bits — chance — down to 6.6 and flattens out without solving it. The
  feed-forward run was stopped at 180k, and the dashed line is drawn rather than measured.
  Compare the paper's Figure 10.</figcaption>
</figure>

| | first below 0.05 bits | windows at exactly zero | run length |
|---|---|---|---|
| NTM, feed-forward | **39,000** | 107 of 172 | 172,000 (stopped early) |
| NTM, LSTM controller | **22,000** | 384 of 500 | 500,000 |
| LSTM | never | 0 of 500 | 500,000 |

Both NTMs drop off a cliff between 20k and 40k sequences. Neither then sits at zero cleanly:
each has occasional spikes for the rest of the run, with the longest unbroken stretch of
exactly-zero windows being about 67,000 sequences for each. I stopped the feed-forward run at
172,000; the other two ran the full 500,000. The LSTM grinds from 18 bits to 6.6 and flattens out
around 300,000 without solving it.
The paper puts its NTM at "near zero cost within approximately 30,000 episodes, whereas LSTM
does not reach zero cost after a million". Mine crosses 0.05 bits at 22,000 and 39,000.

<figure>
  <img src="/assets/ntm-recall-generalisation.png" alt="Cost against number of items per sequence for three models">
  <figcaption>Trained on 2 to 6 items, both NTMs stay under 1.5 bits out to 20 items — better at
  every point measured than the paper's Figure 11, where its best model is at 6.8 bits there. I
  have no explanation for it.</figcaption>
</figure>

Cost per sequence in bits, 200 episodes per point:

| items per sequence | 6 | 10 | 15 | 20 |
|---|---|---|---|---|
| LSTM | 15.05 | 18.43 | 19.39 | 20.34 |
| NTM, feed-forward | **0.00** | **0.00** | **0.10** | 2.33 |
| NTM, LSTM controller | **0.00** | **0.00** | **0.00** | **0.97** |

The paper describes its own feed-forward NTM as "nearly perfect for sequences of up to 12 items
(twice the maximum length used in training)", and "still has an average cost below 1 bit per
sequence for sequences of 15 items". Mine is at 0.10 bits at 15 items and 0.00 at 12, and the
LSTM-controller one is at exactly zero out to 15. At twenty items, well past anything the paper
reports, both of mine are still under 2.5 bits.

So mine generalise further than the paper's on this axis, and I have no explanation. My models
are 16% and 7% *smaller* by parameter count, so it is not capacity, and the settings are the ones
the tables specify. Read those numbers as estimates with real noise: at 15 items a 50-episode
sample gave me 1.77 for the feed-forward model where 200 episodes gives 0.10.

<figure>
  <img src="/assets/ntm-recall-memory.png" alt="Memory use during an associative recall episode">
  <figcaption>Recall uses a different mechanism from copy. The write head still walks a diagonal
  through the episode, but at query time the read head does not scan — it lands directly on the
  location holding the item that followed the query. The colour key is printed on the figure.</figcaption>
</figure>

## New experiment: taking the memory away

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
  <figcaption><strong>New experiment.</strong> Twelve slots hold twenty vectors at zero bit error
  across every length tested (left); error only appears in earnest at eight. The control (right)
  rules out capacity — every memory there holds the same 400 numbers, and error still climbs to a
  third of the bits as the slots get fewer and wider. Dotted lines mark where each memory runs out
  of slots.</figcaption>
</figure>

**Twelve slots is enough for twenty vectors.** One wrong bit in 2,240 at length 14 and exact zero
at the other nine lengths tested, which means eight of the twenty vectors have nowhere of their
own to live. Error only shows up in earnest at eight slots.

A heatmap of where the head wrote cannot tell packing from overwriting, so the test is a linear
probe fitted from each slot's contents to each input vector.

<figure>
  <img src="/assets/ntm-pressure-decodability.png" alt="Grids of memory slot against input position, shaded by linear decodability">
  <figcaption>One bright diagonal at 20 slots — one vector per slot, as the paper describes. Two
  at 12: each slot holds two input vectors and the probe recovers both. At 8 the second diagonal
  fades and a block of the sequence is not recoverable at all. 400 sequences per panel.</figcaption>
</figure>

*Vectors per slot* counts how many input positions a single slot yields; *slots per vector*
counts the reverse, and stays at 1.00 if nothing is smeared across several slots.

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
  of memory, wraps to the start, and lays a second diagonal over the first. At 8 it wraps twice and
  the read head loses the thread. Bottom row: the vectors actually written.</figcaption>
</figure>

### It runs out of addresses, not room

Every memory in the control arm holds the same 400 numbers, arranged differently:

| Memory | Numbers | Parameters | Bit error at length 20 |
|---|---|---|---|
| 20 × 20 | 400 | 13,720 | **0.0%** |
| 12 × 20 | 240 | 13,544 | **0.0%** |
| 8 × 50 | 400 | 29,086 | 24.2% |
| 4 × 100 | 400 | 54,728 | 34.0% |
| 2 × 200 | 400 | 106,024 | 35.1% |

Same storage, and error climbs from nothing to a third of the bits as the slots get fewer and
wider. Width is not a neutral knob on its own — holding slots at 8 and widening them from 20 to
50 also makes things worse, 12.5% to 24.2% — so this arm shows that capacity is not what is
missing, rather than isolating slot count perfectly. **Twelve slots of width 20 hold 240 numbers and score 0.0%. Eight slots
of width 50 hold 400 numbers, cost twice the parameters, and score 24.2%.** More room, worse
result.

Three caveats on this section. It is one training run per configuration, and 16 slots scoring
7.4% where 12 scores 0.0% is the sign of it — that is almost certainly variance, which also means
12 slots scoring zero could be a lucky seed, and the exact breaking point is not pinned down. The
probe has no null: I never fitted it against a vector the model had not seen, which is the
control that would tell me how much of the diagonal is the probe rather than the memory. And copy
forbids slot reuse by construction, because nothing is emitted until the whole input has been
read — that is what makes packing the only available explanation here, and it also means nothing
here says what the network would do on a task where recycling is possible.

## The scorecard

The separation of the curves, zero cost for both NTMs against a baseline that never gets there,
perfect copy generalisation past the training range, the learned algorithm visible in the memory
traces, and convergence on associative recall at 37,000 sequences against the paper's
approximately 30,000 — all of that matched. Three things did not.

**The controller ordering.** The paper says the feed-forward controller "learns faster than NTM
with an LSTM controller", in the associative recall section. On copy mine agrees: 5,000 sequences
against 10,000. On associative recall, the task the claim is actually about, mine reverses it —
22,000 for the LSTM controller against 39,000 for the feed-forward one. One seed each, so I would
not lean on it, but it is the wrong way round on the task the paper makes the claim for.

**Parameter counts**, by 1.3 to 16.2%, as above. I believe this one is the paper's.

**Generalisation on associative recall**, where I come out better than the published result.
Still unexplained.

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

Most of the results here came off rented A10Gs through Modal. The NTM launches a few hundred tiny
kernels per sequence, so a GPU spends its time on launch overhead rather than arithmetic. I
capture the whole training step as a CUDA graph, one per sequence length, which is worth about 8x
on the GPU and is what makes renting one worthwhile at this size.
