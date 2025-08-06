echo "Evaluating bpe, uni, wpc, and TreeTok on Morpho dataset"

##############################
### change these variables ###
##############################
vocab_size=30000
data_dir=data/norm_wiki103
treetok_checkpoint=1w_singlecomp
##############################

for tokenizer_type in bpe uni wpc; do
    if [ ! -d "${data_dir}/ext_vocab_${tokenizer_type}_${vocab_size}" ] || [ -z "$(ls -A ${data_dir}/ext_vocab_${tokenizer_type}_${vocab_size})" ]; then
        echo "Training ${tokenizer_type} tokenizer with vocab size ${vocab_size}"
        python utils/train_hf_tokenizer.py \
        --corpus_path ${data_dir}/wiki.train.tokens \
        --ext_vocab_output_dir ${data_dir}/ext_vocab_${tokenizer_type}_${vocab_size} \
        --vocab_size ${vocab_size} \
        --tokenizer ${tokenizer_type}
    fi
    echo "Evaluating ${tokenizer_type} tokenizer with vocab size ${vocab_size}"
    python experiments/eval_tokenizer.py \
        --corpus_path data/morpho/goldstd_trainset.segments.eng \
        --tokenizer_type ${tokenizer_type} \
        --tokenizer_path ${data_dir}/ext_vocab_${tokenizer_type}_${vocab_size}/${tokenizer_type}_${vocab_size}.json
done

python experiments/eval_tokenizer.py \
--corpus_path data/morpho/goldstd_trainset.segments.eng \
--pure_input_path data/morpho/goldstd_inputs.txt \
--tokenizer_type tree \
--tokenizer_path out/${treetok_checkpoint} \
--ext_vocab_path vocab_mining/${treetok_checkpoint}/vocab_mining_read.out.readable