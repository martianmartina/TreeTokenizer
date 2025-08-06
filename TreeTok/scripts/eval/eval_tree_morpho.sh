echo "Evaluating composition model out/$1 on Morpho dataset"
python experiments/eval_tree.py \
--corpus_path data/morpho/goldstd_trainset.segments.eng \
--config_path out/$1/all_config.json \
--vocab_dir out/$1/ \
--ext_vocab_path data/norm_wiki103/ext_vocab_bpe_10000/bpe_10000.txt.ids \
--output_dir eval/seg/$1 \
--pretrain_dir out/$1/ \
--checkpoint_name model.bin \
--suffix_offset 0 \
--inference_mode parser 