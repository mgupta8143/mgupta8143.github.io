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

Although Neural Turing Machines by Graves et al. is over a decade old, I reproduced this paper to understand the thinking behind memory mechanisms prior to Attention is All You Need coming out in 2017, and how these ideas shaped modern transformers, as well as where they failed to stand the test of time. NTMs in theory sound like a great mechanism for giving controller networks additional memory capabilities, but in practice, training often takes significantly more overhead than SOTA transformers, and often leads to unstable training.

I reproduced the copy and associative recall tasks out of the five in the paper because they were the most checkable: both have published learning curves I could hold mine against, and when they
fail you can look at the output and see it failing. Everything was trained on rented A10G
GPUs through [Modal](https://modal.com) and on my own laptop, and all of it is in the
[code](https://github.com/mgupta8143/neural-turing-machines).

The result of this reproduction held up against the reports made in the paper (detailed further below). When training both tasks, both NTM (feedforward and LSTM) variants reach exactly zero cost on the copy and associative-recall tasks and the LSTM baseline never does. We used the exact same configurations, parameters, etc. to make this reproduction hold up. The paper's central claim is that an NTM should learn an algorithm rather than fit a task when an external differentiable memory is provided. It supports this with its Figure 6, which shows the read and write heads tracing a clean diagonal across memory, and it goes as far as describing the algorithm it believes the network discovered: write each input vector to a successive memory location, then return to the start and read them back in order.


## Setup

**Copy** is the simpler of the two. Show the network a short sequence, then stop feeding it
anything and ask it to say the sequence back. Nothing is available during the recall except what
it put in memory, so the task is a direct test of whether it can store and retrieve.

**Associative recall** asks for indirection instead. Show it a list of items one after another,
then show it one of those items again as a query, and ask for the item that came *next* in the
list. Getting it right means finding the query in memory and then finding what sits beside it.

| | Copy | Associative recall |
|---|---|---|
| One item is | an 8-bit vector | three 6-bit vectors |
| An episode shows | 1 to 20 items | 2 to 6 items |
| Scored steps | the recall phase | the three answer steps |
| Chance | 84 bits | 18 bits |
| Paper section | 4.1 | 4.3 |

Training settings, against the paper:

| | Paper | Mine |
|---|---|---|
| Optimiser | RMSProp, Graves (2013) form, momentum 0.9 | same |
| Learning rate | 1e-4 NTM, 3e-5 LSTM (copy); 1e-4 all (recall) | same |
| Batch size | not stated | 1 |
| Memory | 128 × 20 | same |
| Gradient clipping | elementwise to (−10, 10) | **norm, not elementwise** |
| Training length | 500k+ sequences | 500k |

The clipping row is a real deviation and I explain it below.

The parameter counts don't reconcile, and not by a rounding error:

| | Copy: mine | Copy: paper | Recall: mine | Recall: paper |
|---|---|---|---|---|
| LSTM baseline | 1,328,136 | 1,352,969 | 1,326,598 | 1,344,518 |
| NTM, feed-forward | 16,096 | 17,162 | 123,046 | 146,845 |
| NTM, LSTM controller | 65,696 | 67,561 | 65,054 | 70,330 |

Mine come out consistently smaller. I spent longer than I want to admit looking for a reading of
"4 heads" that makes the numbers come out, and there isn't one.

## Getting it to run at all

The model I mostly understood going in. What ate the evenings was finding somewhere to run it.

I started on Colab overnight, which failed in the most tedious way possible: the runtime
disconnects, and you come back in the morning to nothing. Then I ran it locally, which works, but
a full 500,000-sequence run is hours on a laptop CPU and I wanted to try more than one thing a
day. I ended up on Modal, which rents GPUs by the second and comes with $30 of free credit a
month — enough for this entire project, several times over.

Renting a GPU then did almost nothing, which confused me for a while. The NTM launches a few
hundred tiny kernels per sequence, so the GPU spends its time on launch overhead instead of
arithmetic — run naively it's barely faster than the laptop was. The fix is to capture the whole
training step as a CUDA graph, one per distinct sequence length, since every batch has a
different one. That's about 8x against the eager GPU path, and it's the difference between one
run a day and several. It's the only reason renting a GPU bought me anything.

The code is at
[github.com/mgupta8143/neural-turing-machines](https://github.com/mgupta8143/neural-turing-machines).
The addressing lives in `src/models/ntm/memory.py`, one function per stage of the paper's
Figure 2. If you only open one file, open that one.

## Four omissions that decide whether it trains

Roughly in the order of how much time each one cost me.

**RMSProp as Graves (2013) defines it.** Section 4.6 cites that form without stating it:
centered, decay 0.95, damping 1e-4. PyTorch's defaults are 0.99 and 1e-8, and that 1e-8 inflates
the update wherever gradient variance is small — which is most of an LSTM controller's recurrent
matrix. With the defaults the model reaches zero and then drifts back off it.

**Clip the gradient norm, not each component.** This one is a deviation rather than an omission,
since the paper does specify elementwise clipping. But with the paper's optimiser, pre-clip
gradient norms peak at 566 to 176,000 times their running median, and clipping each component to
(-10, 10) lets a spike like that through as an update large enough to destroy the addressing.

**The starting memory has to break symmetry.** Identical rows means every location gets an
identical gradient, so the 128 of them never differentiate and the model sits near chance.

**Sharpening underflows if written literally.** Equation 9 raises the weighting to a power; small
weights at a high power underflow float32 to zero and the head ends up attending to nothing.
`softmax(γ log w)` is the same expression and survives.

And a fifth, which isn't an omission so much as something the paper had no reason to mention:
**the range of training lengths decides whether it trains at all.** Copy draws its sequence
length from 1 to 20. I tried narrowing that, expecting shorter sequences to be easier, and got
the opposite. Chance differs with the range, so the comparable number is cost as a fraction of
it:

| lengths trained on | cost at 10,000 sequences, as % of chance | solved by 10,000 |
|---|---|---|
| 1 to 20 | **0.0%, 3.1%, 46.8%** (three seeds) | 1 of 3 |
| 1 to 15 | 1.1% (one seed) | yes |
| **1 to 10** | **61.4%, 67.7%, 69.4%** (three seeds) | **0 of 3** |
| fixed length 10 | 76.9% (one seed) | no |

The two arms don't overlap — the worst 1-to-20 seed still beats the best 1-to-10 seed, and no
1-to-10 seed gets below 61% of chance inside the budget. A fixed length is worse again. But only
one of the three 1-to-20 seeds actually solved it in 10,000 sequences, so I'd describe the
separation as real and the speed as not pinned down.

My guess is that the short examples in a wide range are what break the addressing symmetry
cheaply, and everything else bootstraps off them. Narrowing the range removes the easy examples
and there's nothing left to start from. This is also why I ran the memory experiment below on a
range of lengths rather than pinning the sequence — pinning it would have measured this failure
instead of the one I was after.

## Copy: both NTMs reach zero, the LSTM never does

Each model trained for 500,000 sequences at one sequence per update, which is about 100 minutes
on a rented A10G for the NTMs and rather less for the LSTM. I already knew the NTM would win; the
paper told me that. What I wanted was the size of the gap, and whether the LSTM fails the way the
paper describes.

Both came out right. Both NTMs drop to zero inside the first few thousand sequences and the LSTM
spends half a million getting to one bit. Worth saying plainly: that LSTM has 1,328,136
parameters and the feed-forward NTM that beats it has 16,096.

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

Hitting zero on the training distribution is the easy part. The paper's actual claim is that
whatever it learned keeps working outside that range, so the test is to hand a trained model
sequences much longer than anything it saw. Memorisation falls apart the moment you leave the
training lengths. A model genuinely using the memory shouldn't care how long the sequence is.

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

Both NTMs copy perfectly at six times their training length. This is better than the paper's own result: its Figure 4 caption reports "a few more local errors and one global error"
at length 120, where a vector is duplicated and everything after it shifts by one. Mine makes
neither mistake.

So it generalises. Now the *how*, and this is the part of the NTM I actually like: you can just
look. Every read and write leaves a weighting over memory, so you get the addressing timestep by
timestep instead of guessing at it from behaviour.

<figure>
  <img src="/assets/ntm-copy-memory.png" alt="Write and read weightings over time on a length-40 copy">
  <figcaption>The write head lays one sharp diagonal as the input arrives and the read head
  retraces the same path on the way out: +1.00 locations per timestep, weighting pinned at 0.999.
  Length 40, double the training range; the paper's Figure 6 for comparison.</figcaption>
</figure>

Measured on the trained model, the write head advances +1.00 memory locations per timestep with
its weighting pinned at 0.999, and the read head retraces the same path. That is the algorithm
from the top of this page, to the decimal.

## Associative recall: the NTMs solve it, the LSTM never gets close

Copy only needs the network to walk in a straight line. Associative recall needs indirection —
finding a thing, then finding what sits next to it — which is the part the paper argues an
external memory is genuinely better at than an LSTM's internal state. If that is right, the gap
between the NTM and the baseline should be wider here than on copy.

<figure>
  <img src="/assets/ntm-recall-curves.png" alt="Learning curves for three models on associative recall">
  <figcaption>Both NTMs fall off a cliff between 20k and 40k sequences and hold exactly zero.
  The LSTM grinds from 18 bits — chance — down to 6.6 and flattens out without solving it. The
  feed-forward run was stopped at 172k, and the dashed segment after that is drawn, not measured.
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

Generalisation on this task means more items than the network was trained on, so the sweep runs
from six items up to twenty against a training maximum of six. This is the figure where my numbers came
out better than the published ones, which I did not expect.

<figure>
  <img src="/assets/ntm-recall-generalisation.png" alt="Cost against number of items per sequence for three models">
  <figcaption>Trained on 2 to 6 items, both NTMs stay under 1.5 bits out to 20 items — better at
  every point measured than the paper's Figure 11, where its best model is at 6.8 bits there.</figcaption>
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

So mine generalise further than the paper's here, and I can't account for it. The obvious
explanations don't work: my models are 16% and 7% *smaller* by parameter count, so it isn't
capacity, and the settings are the ones the tables specify. Three candidates I can't rule out,
in the order I'd check them. I clip gradients by norm where the paper clips elementwise, and I
documented above that elementwise clipping is what destroys the addressing — so my late-training
stability may simply be better than theirs. I train at batch 1, which the paper never states.
And I read their numbers off Figure 11 rather than from a table, so the two axes may not be
measuring the same thing. I'd want the last one ruled out first. Read those numbers as estimates with real noise: at 15 items a 50-episode
sample gave me 1.77 for the feed-forward model where 200 episodes gives 0.10.

<figure>
  <img src="/assets/ntm-recall-memory.png" alt="Memory use during an associative recall episode">
  <figcaption>Recall uses a different mechanism from copy. The write head still walks a diagonal
  through the episode, but at query time the read head does not scan — it lands directly on the
  location holding the item that followed the query. The colour key is printed on the figure.</figcaption>
</figure>

## New experiment: taking the memory away

Reproducing the paper tells you the network learns the algorithm the paper says it learns. What I
wanted to know was whether that is the *only* algorithm it has, or just the one the task happened
to ask for.

Because the algorithm in the paper is not complicated. Write each vector to the next slot along,
walk back to the start, read them out in order. My traces confirm it exactly — the write head
moves +1.00 slots per timestep with its weighting pinned at 0.999, and the read head retraces the
same path. It is a tape.

It only has to be a tape because the memory is enormous. 128 slots to hold at most 20 vectors is
six times the room the obvious strategy needs, so the model is never under pressure and the
obvious strategy never fails. Every published NTM run I could find is like this. So I wanted to
take the room away and see what came out instead.

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
counts the reverse, and stays at 1.00 if nothing is smeared across several slots. A position
counts as recoverable when the probe's R² exceeds 0.5 — the matrices are close to binary, so the
counts don't move for thresholds anywhere between about 0.3 and 0.8.

| Slots | Vectors per slot | Slots per vector | Positions recoverable |
|---|---|---|---|
| 20 | 1.00 | 1.00 | 20 / 20 |
| 12 | **1.67** | 1.00 | **20 / 20** |
| 8 | 1.25 | 1.00 | 10 / 20 |

At twenty slots there is one bright diagonal: one vector per slot, as the paper describes. At
twelve there are **two** diagonals — slot 1 holds vector 1 *and* vector 13, and both come back
out. 1.67 is exactly 20/12, so every vector is accounted for and the load is even. This is the bit I
was pleased with. I expected to find it dropping vectors at twelve slots, and instead the load
came out exactly even, and I still don't have a reason why it should be even rather than
lopsided. Slots per
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
| 8 × 20 | 160 | 13,456 | 12.5% |
| 8 × 50 | 400 | 29,086 | 24.2% |
| 4 × 100 | 400 | 54,728 | 34.0% |
| 2 × 200 | 400 | 106,024 | 35.1% |

Same storage, and error climbs from nothing to a third of the bits as the slots get fewer and
wider. Width isn't a neutral knob either. Hold the slots at 8 and widen them from 20 to 50 and it also
gets worse, 12.5% to 24.2%. So this arm doesn't isolate slot count cleanly and I'm not going to
pretend it does. What it does rule out is capacity. **Twelve slots of width 20 hold 240 numbers and score 0.0%. Eight slots
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

Most of it matched. The curves separate the way the paper shows them separating. Both NTMs reach
zero cost and the baseline never does. Copy generalises past the training range with no errors.
The algorithm is there in the memory traces. Associative recall converges at about the episode
count the paper gives.

Three things did not match.

And one task isn't here at all. I also reproduced repeat copy, the paper's Section 4.2, and cut
it. It converged, and I found something I liked: the interpolation gate — the switch that decides
whether a head uses content addressing or just carries on shifting — predicted generalisation
cleanly. Every converged run with an open gate beat every run with a closed one, and biasing it
open took the error at twice the training length from 18.5% down to 0.68%. Then I ran more
seeds. The setting that produced that trained on two seeds out of four, and the settings either
side of it trained on none. A result that shows up half the time isn't a reproduction, so the
task came out rather than going in with a caveat attached. The code for it is in the git history
if anyone wants to take it further.

**The controller ordering.** The paper says the feed-forward controller "learns faster than NTM
with an LSTM controller", in the associative recall section. On copy mine agrees: 5,000 sequences
against 10,000. On associative recall, the task the claim is actually about, mine reverses it —
22,000 for the LSTM controller against 39,000 for the feed-forward one. One seed each, so I would
not lean on it, but it is the wrong way round on the task the paper makes the claim for.

**Parameter counts**, by 1.3 to 16.2%, as above. I believe this one is the paper's.

**Generalisation on associative recall**, where I come out better than the published result. My
three candidate explanations are in that section.

## Open questions

**Does it actually read the memory, or has it memorised?** Everything the paper offers is
correlational: weightings that look like an algorithm, outputs that come out right. A model that
kept the sequence in its controller state and drove the heads along a diagonal anyway would
produce an identical Figure 6. The test is an intervention — overwrite a memory location
mid-recall and see whether the output follows. If it reads, the output changes to what you
injected. Cheap to run, since it needs no training.

**Why is my associative recall generalisation better than the paper's?** Smaller models, same
settings, better numbers at every point. I have three candidates and no way to choose between
them without rerunning their configuration, which is the next thing I'd do.

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
