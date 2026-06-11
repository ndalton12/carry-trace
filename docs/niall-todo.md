# TODO

Thursday
* delimiter ablation review?
* Plan to do most on 32b instruct, subset on 32b think

Rest of week
* Make presentation on the paper
* Generate real dataset, upload to Huggingface

Stuff to run
* Run goal 1 for real
* Repeat for different checkpoints

# Misc Notes
When using greedy decoding (do_sample=False), temperature is ignored, which is fine:
"The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details."

Also use api for behavior tests? nevermind - not available
