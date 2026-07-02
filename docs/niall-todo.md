# TODO

Todos
* Analyze activations with probing across layers, generate figures

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
