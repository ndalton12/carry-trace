# Rank, Compressibility, Memorization, and Generalization in Transformers

**A plain-English literature note on recent work from 2023–2026**  
**Last checked:** 20 July 2026

## Executive summary

A growing line of research asks whether the internal representations of a neural network become **simpler, lower-rank, or more compressible** when the network stops memorizing its training set and starts learning a rule that generalizes.

The cleanest evidence comes from **grokking** experiments. In grokking, a model reaches nearly perfect training accuracy while test accuracy remains poor, then—sometimes much later—test accuracy rises sharply. Several papers find that this transition occurs alongside a reduction in effective rank, a compression of activation geometry, or a reduction in the complexity of the function implemented by the network.

This supports an appealing story:

> The model first fits the examples with a relatively complicated solution, then gradually discovers a simpler and more reusable computation.

There is also evidence at language-model scale. Studies of Pythia and OLMo checkpoints find non-monotonic changes in representation geometry: an early collapse, a later expansion associated with short-context or n-gram learning, and then a selective compression associated with improvements in longer-context capabilities.

However, the evidence does **not** support the simple rule:

> Lower rank always means better generalization.

Low rank can mean useful abstraction, but it can also mean representation collapse or underfitting. High rank can mean memorized exceptions, but it can also mean the model has learned many genuinely useful features. Rank is strongly affected by the optimizer, weight decay, batch size, architecture, model width, training duration, and the exact matrix or activation set being measured.

The most defensible current conclusion is:

> **Rank and compressibility are promising diagnostics of changes occurring inside one training run, but they are not reliable standalone scores for comparing the generalization of unrelated language models.**

The **trajectory** of the spectrum across checkpoints is usually more informative than one final rank value.

---

## 1. The idea in plain English

Suppose a model must learn modular addition.

A memorizing solution might effectively store many separate cases:

- \(3 + 4 \mapsto 7\)
- \(3 + 5 \mapsto 8\)
- \(4 + 4 \mapsto 8\)
- and so on.

A generalizing solution instead represents a common rule that applies to all cases.

The first strategy may require many independent internal directions or many distinct local computations. The second may be expressible through a smaller number of reusable features.

This motivates the hypothesis:

> Memorization often corresponds to a more complicated internal description, while generalization often corresponds to a more compressed description.

This idea is related to:

- Occam's razor;
- minimum description length;
- algorithmic complexity;
- information bottleneck arguments;
- low-rank matrix factorization;
- intrinsic dimensionality;
- singular learning theory.

But these concepts are related rather than interchangeable. A paper measuring the rank of a weight matrix is not necessarily measuring the same thing as a paper measuring the intrinsic dimension of token activations.

---

## 2. What “rank” and “compression” can mean

### 2.1 Exact matrix rank

The exact rank of a matrix is the number of linearly independent directions it contains.

For trained neural-network matrices, exact rank is often unhelpful because nearly all singular values are nonzero numerically. A matrix can technically have full rank while most of its behavior is concentrated in a few directions.

### 2.2 Effective rank

Let the singular values of a matrix \(W\) be:

\[
\sigma_1 \geq \sigma_2 \geq \cdots \geq 0.
\]

A common entropy-based effective rank is:

\[
p_i = \frac{\sigma_i}{\sum_j \sigma_j},
\qquad
\operatorname{erank}(W)
=
\exp\left(-\sum_i p_i \log p_i\right).
\]

Interpretation:

- If one singular value dominates, effective rank is close to 1.
- If \(k\) singular values contribute equally, effective rank is close to \(k\).
- A decline means that the matrix is concentrating its action into fewer dominant directions.

Some papers normalize effective rank by the largest possible rank.

### 2.3 Weight rank

A paper may calculate effective rank for:

- attention projection matrices;
- MLP matrices;
- embedding matrices;
- the unembedding/output matrix;
- all weight matrices in the network.

Weight rank measures the structure of the parameterized linear maps. It includes directions that may rarely or never be used on the data being evaluated.

### 2.4 Activation or feature rank

Collect model activations for a dataset into a matrix:

\[
Z \in \mathbb{R}^{N \times d},
\]

where rows are examples or tokens and columns are hidden features.

The spectrum of \(Z\), or of its covariance/correlation matrix, measures the dimensionality of the representation the model actually uses on those inputs.

Activation rank can be more behaviorally meaningful than weight rank, but it depends on:

- the input dataset;
- which tokens are sampled;
- which layer is measured;
- whether activations are centred or normalized;
- sequence length and sample count.

### 2.5 Intrinsic dimension

Intrinsic dimension asks how many coordinates are locally needed to describe the representation manifold.

A curved two-dimensional surface embedded in a 1,000-dimensional space can have intrinsic dimension 2 even if it is not contained in one two-dimensional linear subspace.

Intrinsic dimension therefore captures nonlinear geometry that ordinary matrix rank misses.

### 2.6 Spectral shape

A scalar effective-rank number discards information. Other useful measurements include:

- the complete singular-value or covariance spectrum;
- the rate at which eigenvalues decay;
- the existence of large outlier directions;
- the stability of singular vectors over time;
- alignment of singular vectors between layers;
- isotropy or anisotropy.

Two models can have the same effective rank while organizing information very differently.

### 2.7 Functional complexity

Some work measures the complexity of the function rather than the dimensionality of one matrix.

The **linear mapping number**, for example, estimates how many distinct local linear computations a nonlinear network implements over its input space.

### 2.8 Practical compressibility

A model may be compressed using:

- low-rank factorization;
- singular-value truncation;
- quantization;
- pruning;
- parameter sharing.

A practical compressibility score asks how much compression is possible before loss or downstream performance deteriorates beyond a selected tolerance.

This is not identical to effective rank. A full-rank matrix may quantize well, while a matrix with a concentrated spectrum may still contain low-energy directions that are essential for a particular task.

---

# 3. Core papers

## 3.1 Grokking, Rank Minimization and Generalization in Deep Learning

**David Yunis, Kumar Kshitij Patel, Samuel Wheeler, Pedro Henrique Pamplona Savarese, Gal Vardi, Karen Livescu, Michael Maire, and Matthew R. Walter — 2024**

- [OpenReview record](https://openreview.net/forum?id=6NHnsjsYXH)
- Status: ICML 2024 Mechanistic Interpretability Workshop paper

### Question

Does the transition from memorization to generalization during grokking coincide with the model finding lower-rank weights?

### Method

The authors train small neural networks, including transformers, on algorithmic grokking tasks and track the singular values of their weight matrices throughout training.

### Main finding

The drop in validation loss during grokking coincides with the model discovering low-effective-rank solutions across its weight matrices.

The model can already have near-zero training loss before this happens. Rank therefore exposes continued internal reorganization during a period when the ordinary training metric appears to have finished changing.

### Plain-English interpretation

The network first finds a way to fit the observed equations. Continued optimization then finds a more economical representation of the rule.

### Why it is important

This is the closest direct match to the proposition:

> “Generalization appears when the internal solution becomes low-rank.”

### Main caveats

- The main tasks are small and synthetic.
- Modular arithmetic has an unusually compact true rule.
- Weight decay is important to the effect.
- Simultaneous timing does not establish that rank reduction causes generalization.

### Evidence strength

**Strong within toy grokking; weak for ordinary LLM generalization.**

---

## 3.2 Deep Grokking: Would Deep Neural Networks Generalize Better?

**Simin Fan, Razvan Pascanu, and Martin Jaggi — 2024**

- [arXiv abstract](https://arxiv.org/abs/2405.19454)
- [arXiv HTML](https://arxiv.org/html/2405.19454)
- Status: preprint

### Question

Does the relationship between rank and grokking also appear in deeper networks, and is activation rank more informative than weight norm?

### Method

The authors train MLPs as deep as approximately 12 layers and track the numerical rank of internal feature matrices.

### Main findings

- Deep MLPs can exhibit grokking.
- Some deep networks show more than one stage of test improvement.
- The beginning of generalization aligns with declining feature rank.
- Multi-stage generalization can align with a double-descent-like trajectory in feature rank.
- The relationship persists across changes in training-set size and weight decay.

### Plain-English interpretation

The hidden representations simplify as the model begins to generalize. In some runs, representation complexity falls, rises, and falls again while the model moves through multiple stages of generalization.

### Why it is important

It shifts attention from parameter norms to **the dimensionality of the features actually produced by the network**.

Weight norm tells us how large the weights are. Feature rank tells us how many independent directions the representation is using.

### Main caveats

- The networks are MLPs, not large autoregressive transformers.
- The authors explicitly leave tests on transformers and other architectures for future work.
- Numerical feature rank can be sensitive to thresholds and data sampling.

### Evidence strength

**Good evidence that activation-rank trajectories track grokking in controlled deep networks.**

---

## 3.3 Approaching Deep Learning through the Spectral Dynamics of Weights

**David Yunis et al. — 2024**

- [arXiv abstract](https://arxiv.org/abs/2408.11804)
- [arXiv HTML](https://arxiv.org/html/2408.11804)
- Status: preprint/OpenReview paper

### Question

Are low-rank spectral dynamics specific to grokking, or do they also occur in ordinary deep-learning systems?

### Method

The paper studies singular values and singular vectors across:

- convolutional image classifiers;
- UNets for image generation;
- LSTMs for speech recognition;
- a roughly 160M-parameter transformer trained on WikiText-103;
- modular-addition grokking;
- true-label and random-label training.

### Main findings

Across several architectures, larger singular values tend to grow disproportionately, lowering effective rank during training.

Weight decay strengthens this spectral concentration beyond simply reducing the overall norm.

In a controlled CIFAR-10 experiment:

- networks trained on true labels have lower-rank middle layers;
- networks trained on random labels retain substantially higher-rank middle layers;
- true-label networks show more persistent alignment between singular directions of adjacent layers.

### Plain-English interpretation

When the labels contain shared structure, the network can reuse common directions. Random labels contain no coherent class rule, so fitting them requires a less shared and more complicated internal solution.

### Why it is important

This is the main bridge from small grokking tasks toward practical architectures. It shows that spectral concentration is not exclusive to one toy transformer.

### Main caveats

- The clean true-label versus random-label comparison uses a small MLP on CIFAR-10, not an LLM.
- Practical tasks do not show the same sharp rank transition as grokking.
- The transformer experiment establishes that the spectral effect exists in language modelling, not that it predicts broad language-model generalization.
- The models remain much smaller than contemporary frontier LLMs.

### Evidence strength

**Broad evidence that spectral concentration is common; only suggestive evidence that it is a universal generalization marker.**

---

## 3.4 Grokking as Compression: A Nonlinear Complexity Perspective

**Ziming Liu, Ziqian Zhong, and Max Tegmark — 2023**

- [arXiv abstract](https://arxiv.org/abs/2310.05918)
- [arXiv HTML](https://arxiv.org/html/2310.05918)
- Status: preprint

### Question

Can grokking be understood as compression of the implemented function rather than only compression of weights or activations?

### Method

The paper introduces the **linear mapping number (LMN)**.

For a ReLU network, different regions of input space can correspond to different local linear maps. LMN generalizes this idea to other activation functions by estimating how many locally distinct mappings are required across sampled inputs.

### Main findings

On modular addition, a permutation-group task, and multi-digit XOR:

- LMN declines steadily after memorization and before generalization;
- LMN has a comparatively simple relationship with test loss;
- ordinary \(L_2\) parameter norm has a more complicated relationship with test loss;
- LMN exposes a switch between different generalizing solutions in an XOR experiment.

### Plain-English interpretation

The network gradually stops using many separate local rules and moves toward a smaller set of shared computations.

### Why it is important

This paper clarifies that “compression” need not mean only low-rank matrices. The central object may be the complexity of the computation performed by the network.

### Main caveats

- LMN is expensive to estimate.
- It depends on the sampled input space.
- The tasks and models are small.
- Scaling the method to natural-language distributions and large transformers is nontrivial.

### Evidence strength

**Conceptually strong, but not yet a practical LLM metric.**

---

## 3.5 The Geometry of Hidden Representations of Large Transformer Models

**Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, and Alberto Cazzaniga — NeurIPS 2023**

- [arXiv abstract](https://arxiv.org/abs/2302.00294)
- [arXiv HTML](https://arxiv.org/html/2302.00294)

### Question

How does the intrinsic dimension of representations change across the layers of large self-supervised transformers?

### Method

The authors estimate intrinsic dimension and neighbourhood structure in transformer models trained on protein sequences and images, including ESM-2 and iGPT models. They also include preliminary analysis of Llama 2.

### Main findings

A common pattern appears across models:

1. early layers expand the data into a higher-dimensional space;
2. intermediate layers compress the representation;
3. later layers maintain or re-expand dimensionality for reconstruction/prediction.

The intermediate low-intrinsic-dimension region often contains the strongest abstract or semantic information, such as protein homology or image-class information.

Checkpoint analysis suggests that the early expansion forms first, followed later by compression in the semantically rich middle region.

### Plain-English interpretation

The model appears to first separate and rearrange the data, then compress it into a lower-dimensional abstract representation.

This resembles an encoder-decoder computation:

- expand to make distinctions easier;
- compress to retain the meaningful structure;
- expand or decode to produce the output.

### Why it is important

It shows that useful abstraction is associated with **layerwise expansion followed by compression**, rather than monotonically decreasing rank everywhere.

### Main caveats

- Most experiments are on protein and image transformers.
- Low intrinsic dimension is associated with semantic content at particular layers, not with whole-model test generalization in general.
- Intrinsic dimension depends on the estimator, representation pooling, and dataset.

### Evidence strength

**Strong evidence for structured expansion-compression profiles across transformer depth; indirect evidence about memorization.**

---

## 3.6 Memorization in Language Models through the Lens of Intrinsic Dimension

**Stefan Arnold — 2025**

- [arXiv abstract](https://arxiv.org/abs/2506.09591)
- Status: preprint

### Question

Are some training sequences more likely to be memorized because of the geometry of their representations?

### Method

The paper estimates the intrinsic dimension of contextualized sequence representations and compares it with empirical memorization rates, while controlling for model size and duplication frequency.

### Main findings

For sufficiently large models and low-to-moderate duplication:

- higher-intrinsic-dimension sequences are generally less likely to be memorized;
- duplication and model scale remain major predictors;
- at very high duplication, memorization saturates and intrinsic dimension matters less;
- small models can show a different or reversed relationship.

### Plain-English interpretation

A structurally complicated sequence may be difficult to store exactly after only a few exposures. A low-dimensional or highly regular sequence may be easier to reproduce verbatim.

But repeated exposure can overwhelm this effect: show the sequence often enough and the model can memorize it regardless of geometry.

### Important distinction

This paper measures the geometry of **individual sequences in representation space**, not the effective rank of the model's weights.

It therefore addresses:

> “Which examples are likely to be memorized?”

rather than:

> “Has the model as a whole discovered a generalizing solution?”

### Main caveats

- The relationship changes with scale and duplication.
- Intrinsic dimension is a proxy for structural complexity, not a direct measure of causal memorization difficulty.
- Results should not be converted into a universal “high dimension is good” rule.

### Evidence strength

**Useful direct evidence linking representation geometry to verbatim memorization, with important conditioning variables.**

---

## 3.7 Tracing the Representation Geometry of Language Models from Pretraining to Post-training

**Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, and Blake A. Richards — 2025**

- [arXiv abstract](https://arxiv.org/abs/2509.23024)
- [arXiv HTML](https://arxiv.org/html/2509.23024)
- Status: preprint/workshop work

### Question

How does the spectrum of LLM activations change during real autoregressive pretraining and post-training?

### Method

The paper measures spectral metrics—including an entropy-based effective-rank measure called **RankMe**—on checkpoints from:

- Pythia models from small scales up to 12B parameters;
- OLMo/OLMo-2 models;
- post-training stages including supervised fine-tuning, preference optimization, and reinforcement-learning-based training.

### Main finding: three pretraining phases

The authors report a non-monotonic sequence:

#### 1. Warmup or collapse

Representations rapidly contract during early learning-rate warmup. Outputs can be repetitive and weakly contextual.

#### 2. Entropy-seeking expansion

Representation dimensionality expands, reportedly by multiple times in some settings. This phase coincides with increased learning of short-context and n-gram statistics.

#### 3. Compression-seeking consolidation

The spectrum becomes more anisotropic: variance is selectively retained along some directions while other directions contract. This phase correlates with improvements in longer-context understanding and downstream performance.

### Plain-English interpretation

The model does not simply become lower-rank as it improves.

A more plausible trajectory is:

1. collapse from a poorly organized initialization;
2. open up many directions to capture varied local patterns;
3. consolidate those patterns into a more structured representation.

### Particularly important result

The authors find that retaining only a small number of top principal components severely damages question-answering performance. Removing some dominant components can have much less effect.

This implies that useful language information is distributed across the spectrum. “The top directions contain everything” is false in these experiments.

### Post-training result

Different post-training methods induce different geometric shifts. The paper associates expansion-like changes with fitting instructional or preference datasets and compression-like changes with reward-aligned consolidation, sometimes with reduced diversity.

### Main caveats

- The authors explicitly describe the main capability relationships as correlational.
- RankMe depends on the sampled activations and evaluation corpus.
- The interpretation of n-gram behaviour as memorization is broader than exact training-data extraction.
- The phases may recur rather than happen exactly once.

### Evidence strength

**Probably the most directly relevant large-language-model evidence in this set, but still correlational.**

---

## 3.8 Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory

**Einar Urdshals, Edmund Lau, Jesse Hoogland, Stan van Wingerden, and Daniel Murfet — 2025**

- [arXiv abstract](https://arxiv.org/abs/2510.12077)
- [arXiv HTML](https://arxiv.org/html/2510.12077)
- Status: preprint

### Question

Is practical neural-network compressibility related to a theoretically grounded measure of model complexity?

### Method

The paper connects:

- minimum description length;
- singular learning theory;
- the local learning coefficient;
- practical compression by quantization and factorization.

Experiments use Pythia checkpoints up to approximately 6.9B parameters.

### Main findings

- Models with larger estimated local learning coefficients tend to be less compressible.
- Quantization compressibility has a particularly close relationship with the complexity estimate over substantial parts of training.
- In some regions, the relationship is approximately linear.

### Plain-English interpretation

If many parameter variations produce almost the same function and loss, the model has redundant degrees of freedom. That degeneracy can make the model easier to encode or compress.

The paper gives mathematical support to the informal idea:

> A model that can be heavily compressed without losing performance is implementing a less complex effective solution.

### Why it matters for generalization

Minimum description length has a deep theoretical connection to generalization: shorter effective descriptions are often favoured because they encode reusable structure rather than arbitrary detail.

However, this paper primarily validates compressibility as a measure of **complexity**, not as a universal predictor of downstream or out-of-distribution performance.

### Main caveats

- Correlation with a complexity measure is not the same as correlation with generalization.
- Compressibility depends on the allowed performance loss and the compression algorithm.
- Different tasks may rely on small, low-energy directions that a global compression score overlooks.

### Evidence strength

**Strong support for “compressibility measures effective complexity”; incomplete support for “compressibility predicts generalization.”**

---

## 3.9 Explaining Grokking in Transformers through the Lens of Inductive Bias

**Jaisidh Singh, Diganta Misra, and Antonio Orvieto — 2026**

- [arXiv abstract](https://arxiv.org/abs/2602.06702)
- [arXiv HTML](https://arxiv.org/html/2602.06702)
- Status: preprint

### Question

Why do architectural and optimization choices change how quickly a transformer groks, and does feature compression remain aligned with generalization under these interventions?

### Method

The authors vary:

- the position of LayerNorm or RMSNorm;
- learning rate;
- weight decay;
- readout scale;
- other inductive-bias-related settings.

They measure the effective rank and eigenspectrum of a pre-logit feature-correlation matrix on modular addition.

### Main findings

- Layer-normalization placement strongly changes grokking speed.
- Feature representations evolve continuously throughout training.
- Configurations that generalize earlier also become compressible earlier.
- Similar compression-generalization alignment appears when weight decay is varied.
- A simple “lazy training suddenly becomes feature learning” explanation is inadequate in their setting.

### Plain-English interpretation

The model is not necessarily doing nothing internally during the long memorization plateau. Its features may be evolving continuously, with generalization becoming visible once they reach a sufficiently structured or compressed state.

### Why it is important

The compression signal survives several interventions. This is more convincing than observing one correlation under one training recipe.

At the same time, those interventions expose a problem: architecture and optimization affect both rank and performance. This makes rank a potential **mediator or symptom** rather than an independent predictor.

### Main caveats

- The experiments remain small modular-arithmetic tasks.
- The measured matrix is a particular pre-logit feature correlation, not the entire transformer.
- The evidence remains correlational across interventions rather than a clean direct manipulation of rank alone.

### Evidence strength

**Good recent evidence for compression as a progress measure in transformer grokking; not yet proof at LLM scale.**

---

## 3.10 Disentangling Geometry, Performance, and Training in Language Models

**Atharva Kulkarni, Jacob Mitchell Springer, Arjun Subramonian, and Swabha Swayamdipta — 2026**

- [arXiv abstract](https://arxiv.org/abs/2602.20433)
- [arXiv HTML, version 2](https://arxiv.org/html/2602.20433v2)
- Status: preprint

### Question

Does effective rank reliably predict the performance of language models when training choices are varied in a controlled way?

### Method

The authors train 108 OLMo-style language models while varying factors such as:

- model size;
- token budget;
- batch size;
- weight decay;
- learning rate.

They study the effective rank of the unembedding matrix, final-layer representations, isotropy, angular geometry, in-distribution loss, out-of-distribution loss, quantization, and fine-tuning behaviour.

### Main findings

- High effective rank sometimes accompanies good performance, but not universally.
- Some low-rank models perform well.
- Low rank is not sufficient to produce late-stage degradation.
- Weight decay and batch size strongly affect geometry.
- In-distribution loss predicts out-of-distribution loss better than effective rank.
- Related geometric metrics do not consistently rescue the prediction.
- Rank is better treated as a training diagnostic than as a model-selection score.

### Plain-English interpretation

Rank often records **how the model was trained**, not only how well it generalized.

Imagine two runners with different heart rates because one is at altitude and the other is at sea level. Heart rate may still be useful within one runner's training history, but comparing the raw number across the two runners can be misleading.

Effective rank can behave similarly across different model recipes.

### Why it is crucial

This paper is the strongest counterweight to an overly enthusiastic interpretation of the positive results.

It does not show that geometry is useless. It shows that:

> geometry must be interpreted conditional on architecture, data, optimization, and the exact object measured.

### Main caveats

- The controlled models are relatively small.
- The main focus is the unembedding matrix and final-token final-layer representations.
- Intermediate layers might contain more informative geometric signals.
- The study remains observational rather than a perfect causal manipulation of geometry.

### Evidence strength

**Strong evidence against using one absolute effective-rank number as a universal LLM generalization metric.**

---

# 4. How the findings fit together

The papers can appear contradictory:

- Grokking papers: **rank falls when generalization begins**.
- The 2025 LLM geometry paper: expansion is associated with n-gram learning, followed by compression and better long-context behaviour.
- The 2026 controlled OLMo study: **high rank sometimes accompanies better models**, and absolute rank is heavily confounded.

These results can coexist because the papers measure different things under different conditions.

## 4.1 Different objects are being measured

| Study type | Object measured |
|---|---|
| Grokking rank minimization | Weight matrices across a small network |
| Deep Grokking | Layer activation/feature rank |
| Transformer inductive-bias study | Pre-logit feature-correlation spectrum |
| LLM geometry tracing | Covariance spectrum of sampled hidden representations |
| Controlled OLMo study | Primarily the unembedding matrix and final-layer representations |
| Intrinsic-dimension papers | Nonlinear dimension of representation manifolds |
| LMN paper | Number of distinct local computations |
| MDL/SLT paper | Practical model compressibility and loss-landscape complexity |

There is no reason these quantities must move in exactly the same direction.

## 4.2 Different comparisons are being made

There are at least three distinct experimental questions:

### Within one training run

Does a metric change shortly before or during a generalization transition?

This is where rank and compression look most promising.

### Across matched runs

Does the metric distinguish runs using the same model and data but different seeds or modest hyperparameter changes?

The evidence is mixed but still potentially useful.

### Across unrelated models

Can one compare rank values across different architectures, scales, data mixtures, and optimizers?

The evidence is currently poor.

## 4.3 Compression can be useful or destructive

Useful compression removes redundant directions while preserving task-relevant structure.

Destructive compression removes information the model needs.

Therefore:

\[
\text{lower rank} \neq \text{better model}
\]

The relevant question is:

> Which directions were suppressed, which were preserved, and what computation changed?

## 4.4 Expansion can be necessary before compression

Several transformer-geometry results suggest an expansion-compression sequence:

1. create enough directions to separate varied patterns;
2. reorganize those directions;
3. consolidate shared structure.

This is analogous to feature engineering:

- first build a rich dictionary;
- then discover which combinations are reusable.

Therefore a temporary increase in rank is not evidence of failure or memorization alone.

---

# 5. Is rank/compressibility a promising correlate of generalization?

## My assessment

### Promising uses

Rank and compression metrics are promising for:

1. **Tracking phase transitions within a run.**  
   They can reveal continued internal learning after training loss has flattened.

2. **Comparing true-label and random-label solutions.**  
   Random labels provide a useful control because they remove shared target structure.

3. **Locating semantically rich layers.**  
   Local minima in intrinsic dimension or other spectral transitions may identify useful representations.

4. **Detecting representation collapse or expansion.**  
   Sudden geometric changes can expose training dynamics not visible in scalar loss.

5. **Generating mechanistic hypotheses.**  
   Spectral changes can point researchers toward particular layers, matrices, or training periods for deeper analysis.

6. **Estimating effective model complexity.**  
   Practical compressibility has credible theoretical support as a complexity measure.

### Weak uses

The metrics are currently weak for:

1. **Selecting the best model across unrelated training recipes.**
2. **Predicting out-of-distribution performance without behavioural evaluation.**
3. **Deciding when to stop training based only on falling rank.**
4. **Inferring exact memorization from one matrix's spectrum.**
5. **Claiming causality from temporal correlation.**
6. **Reducing a whole LLM to one scalar “generalization rank.”**

---

# 6. What a convincing causal study would need

A stronger experiment would manipulate rank or compressibility while holding other factors fixed.

For example:

1. Train multiple transformers with the same architecture, data order, optimizer, and number of steps.
2. At selected checkpoints, intervene on one activation or weight spectrum:
   - truncate selected singular directions;
   - penalize spectral entropy;
   - encourage rank expansion;
   - rotate the representation while preserving singular values;
   - compress only specified layers.
3. Retrain or continue training after the intervention.
4. Measure:
   - in-distribution loss;
   - compositional generalization;
   - length generalization;
   - contamination-controlled benchmark performance;
   - exact-sequence extraction;
   - membership inference;
   - robustness to distribution shift.
5. Compare interventions with the same parameter norm and similar training loss.
6. Test whether restoring the removed directions restores the lost capability.

The strongest result would show that a targeted spectral change predictably changes generalization while ordinary confounds remain controlled.

---

# 7. Recommended measurement protocol for LLM research

For a practical experiment, I would avoid choosing one rank metric in advance.

## 7.1 Track several objects

At each checkpoint, measure:

- effective rank of attention and MLP weights;
- effective rank of the unembedding matrix;
- activation covariance spectra at several layers;
- intrinsic dimension at selected layers;
- singular-vector alignment across checkpoints;
- post-hoc quantization and factorization tolerance.

## 7.2 Use fixed activation datasets

Use several held-out corpora:

- ordinary in-distribution text;
- long-context documents;
- algorithmic/compositional tasks;
- duplicated and unique sequences;
- paraphrases of training-like facts;
- a distribution-shift corpus.

Keep sequence length and sampling procedure fixed.

## 7.3 Track the complete spectrum

Record:

- effective rank;
- stable rank;
- top-\(k\) explained variance;
- spectral entropy;
- power-law or tail fits;
- changes in individual singular directions.

Do not discard the full spectrum after calculating one scalar.

## 7.4 Measure behaviour independently

Compression metrics should be correlated against explicit outcomes:

- exact memorization/extraction;
- membership inference;
- n-gram frequency alignment;
- held-out perplexity;
- compositional accuracy;
- long-context accuracy;
- transfer and OOD loss;
- robustness after fine-tuning.

## 7.5 Compare like with like

The safest comparisons are:

- the same architecture;
- the same data;
- the same tokenizer;
- the same optimizer family;
- matched batch size and weight decay;
- adjacent checkpoints or controlled ablations.

## 7.6 Analyse residual signal

Fit out obvious confounds such as:

- parameter count;
- token count;
- training loss;
- batch size;
- weight decay;
- learning rate;
- model width.

Then ask whether geometry explains any remaining variation in generalization.

---

# 8. A compact mental model

A good way to think about the literature is:

```text
Untrained / early model
        |
        v
Initial collapse or poorly organized representation
        |
        v
Expansion: many directions are opened to fit diverse local patterns
        |
        v
Reorganization: directions become aligned and shared
        |
        v
Selective compression: redundant directions shrink
        |
        v
A reusable rule or abstraction becomes behaviourally visible
```

This is a hypothesis, not a universal law.

Different tasks can skip, repeat, or reverse stages. A model can also collapse too far and lose useful information.

---

# 9. Bottom line

The research programme is credible and worth pursuing, but the useful object is probably not a single number called “rank.”

The most promising signal is a **structured change in spectral geometry**:

- measured over time;
- localized by layer and matrix;
- compared under matched training conditions;
- linked to independent memorization and generalization tests.

A concise summary is:

> **Compression appears to accompany generalization when it represents the consolidation of previously learned features into reusable structure. Compression by itself is neither necessary nor sufficient for generalization.**

For LLMs, the evidence currently favours using rank and compressibility as **mechanistic diagnostics and progress measures**, not as universal performance predictors.

---

# 10. Suggested reading order

## Minimal five-paper path

1. [Grokking, Rank Minimization and Generalization in Deep Learning](https://openreview.net/forum?id=6NHnsjsYXH)  
   The direct rank-minimization claim.

2. [Approaching Deep Learning through the Spectral Dynamics of Weights](https://arxiv.org/abs/2408.11804)  
   Extension across architectures and true versus random labels.

3. [Tracing the Representation Geometry of Language Models from Pretraining to Post-training](https://arxiv.org/abs/2509.23024)  
   The most relevant large-language-model checkpoint study.

4. [Disentangling Geometry, Performance, and Training in Language Models](https://arxiv.org/abs/2602.20433)  
   The strongest warning about confounding and absolute rank scores.

5. [Compressibility Measures Complexity](https://arxiv.org/abs/2510.12077)  
   The theoretical bridge from practical compression to effective complexity.

## Add these for a broader view

- [Deep Grokking](https://arxiv.org/abs/2405.19454) — activation rank in deep networks.
- [Grokking as Compression](https://arxiv.org/abs/2310.05918) — function-level complexity.
- [The Geometry of Hidden Representations of Large Transformer Models](https://arxiv.org/abs/2302.00294) — layerwise intrinsic dimension.
- [Memorization in Language Models through the Lens of Intrinsic Dimension](https://arxiv.org/abs/2506.09591) — geometry of examples and verbatim memorization.
- [Explaining Grokking in Transformers through the Lens of Inductive Bias](https://arxiv.org/abs/2602.06702) — recent controlled transformer-grokking evidence.

---

## Bibliographic caution

Several 2025–2026 entries are recent preprints. Their claims, versions, and eventual publication venues may change. The summaries above distinguish what the papers directly demonstrate from broader interpretations of those results.
