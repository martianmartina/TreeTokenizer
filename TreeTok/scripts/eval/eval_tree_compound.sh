echo "Evaluating composition model out/$1 on Compound dataset"
python experiments/eval_tree.py \
--corpus_path data/compound/valid.csv \
--config_path out/$1/all_config.json \
--vocab_dir out/$1 \
--ext_vocab_path data/norm_wiki103/ext_vocab_bpe_10000/bpe_10000.txt.ids \
--suffix_offset 0 \
--output_dir eval/compound/$1 \
--pretrain_dir out/$1 \
--checkpoint_name model.bin \
--inference_mode parser \
--lang en
