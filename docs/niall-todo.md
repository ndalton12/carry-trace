# TODO

Todos
* Goal 4 stuffs
* Update goal2 figure to be more readable

Models to use
* Instruct no RL: Olmo-3-7b-Instruct-SFT
* Instruct + RL: Olmo-3-7b-Instruct

# Misc Notes
* When using greedy decoding (do_sample=False), temperature is ignored, which is fine:
  "The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details."

* Also use api for behavior tests? nevermind - not available
* Think SFT is concise; Think-RLVR is verbose in its thinking, often double checking. Probably not trained for token efficiency, instead its rewarded for correctness only. More likely frontier models are trained for thinking efficiency. We can verify this for Olmo directly.
* Addition as a microcosm for computational behaviour within the CoT (e.g. kill-chain, vaccine distribution, etc)
* Think RLVR is verbose and often commits early but keeps re-checking its answer ("wait, actually...")

# Experiment Runtime Data
* 32b instruct 2048 tokens, takes about 0.8 mins for 18 examples
* 32b think 4096 tokens, takes about 24 mins for 18 examples, force close 50%
* 32b instruct has a very hard time processing the delimited case properly

# More notes
* Circuit overlap? Which areas most active? Not causal necessarily nor do we know what they are doing without circuit analysis
* Need to know the role of CoT tokens. Regenerate partial CoT or just ablate sentences (thought anchors)?
* Unclear how internal state changes during CoT (dynamical system view?) SFT derives answer then explanation while Full uses CoT as computation? Maybe. Also compression view of internals during the computation could be interesting (rank of activations?)
* Ablate sentences and model sub spaces together? Could be interesting. Which sentences are important to which sub spaces. Consider neuron level as subspaces as well (in addition to circuits or residual stream etc)
* Which model internals are important to which tokens? What are those internals doing? How does this evolve across the CoT? Main questions.
* Look at the reasoning theatre plus early exits papers (amazon one at ICML) for inspiration (plus posters in photos on phone) 
  * https://arxiv.org/pdf/2603.05488 Reasoning Theater
  * https://arxiv.org/pdf/2604.18419v6 Knowing When to Quit
  * https://openreview.net/pdf?id=IUX6hpNZBd Backtracking is Decorative
* There is this idea that base models know all this information already but are not organized to elicit it properly and RL does this. This implies that the same CoT between sft and full should be more useful to full. Actually does line up with probe results from goal 2?
