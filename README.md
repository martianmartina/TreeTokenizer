# TreeTokenizer
Hierarchical tokenization with unsupervised morphological structures.

Check out our paper:  
**[Unsupervised Morphological Tree Tokenizer](https://aclanthology.org/2025.findings-acl.1146.pdf)**  
Accepted at ACL 2025 Findings

## Setup
1. Create the conda environment using `environment.yml`.

2. Compile C++ codes.

    ```bash
    python setup.py build_ext --inplace
    ```

## Composition Model Pretraining
1. Download pretraining corpus. 

    We use the training dataset of WikiText-103: https://huggingface.co/datasets/Salesforce/wikitext. For our main experiment in the paper, we use the normalizer in `bert-base-uncased` to preprocess the corpus.

2. Build the heuristic base vocabulary and start the pretraining.
    ```bash
    bash scripts/base_pretrain.sh [NUM_GPUS]
    ```

## Vocabulary Construction
1. Use the pretrained composition model to construct a vocabulary for a specific downstream dataset. For example,
    ```bash
    python utils/vocab_mining_with_tree.py \
    --model_name 1w_singlecomp \
    --ckpt_dir out \
    --data_path data/norm_wiki103/wiki.train.tokens \
    --vocab_size 30000 \
    --continuing_subword_prefix "##"
    ```

Now, we can initialize a TreeTok tokenizer having both the composition model and the constructed vocabulary. The relevant code for initialization and segmentation is inside `utils/tree_tokenizer.py`.
