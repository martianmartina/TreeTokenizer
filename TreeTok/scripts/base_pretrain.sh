if [ ! -d "data/norm_wiki103/ext_vocab_bpe_10000" ] || [ -z "$(ls -A data/norm_wiki103/ext_vocab_bpe_10000)" ]; then
    python utils/train_hf_tokenizer.py \
    --corpus_path data/norm_wiki103/wiki.train.tokens \
    --ext_vocab_output_dir data/norm_wiki103/ext_vocab_bpe_10000 \
    --vocab_size 10000 \
    --tokenizer bpe
fi

torchrun --standalone --nnodes=1 --nproc_per_node=$1 trainer/fast_r2d2_io_ext_vocab_pretrain.py \
--max_grad_norm 1.0 --lr 5e-4 --parser_lr 1e-2 \
--config_path config/fast_r2d2_io_char/config.json \
--corpus_path data/norm_wiki103/wiki.train.tokens \
--preload_vocab_path data/norm_wiki103/ext_vocab_bpe_10000/basic_vocab.txt \
--ext_vocab_path data/norm_wiki103/ext_vocab_bpe_10000/bpe_10000.txt \
--output_dir out/1w_singlecomp \
--batch_size 128 \
--max_seq_len 512 \
--epochs 1 \
--log_step 30 \
--save_steps 6000 \
--sampling_times 10 \
--span_objective allspans \
--no_continuing_prefix \
--tie_decoder