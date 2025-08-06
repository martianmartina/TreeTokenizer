import os
import argparse
import sys
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

cmd = argparse.ArgumentParser()
cmd.add_argument('--corpus_path', type=str, default='data/wikitext-103/wiki.train.tokens')
cmd.add_argument('--ext_vocab_output_dir', type=str, default=None, required=False)
cmd.add_argument('--vocab_size', type=int, required=True)
cmd.add_argument('--tokenizer', type=str, default='bpe', choices=['bpe', 'uni', 'wpc'])
cmd.add_argument('--continuing_subword_prefix', action='store_true', default=False, help='bpe-only arg')
args = cmd.parse_args(sys.argv[1:])
print(vars(args))

unk_token = "[UNK]"
spl_tokens = ["[PAD]","[UNK]", "[BOS]", "[EOS]", "[MASK]"]
input_files = [args.corpus_path]

if args.tokenizer == 'bpe':
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    if args.continuing_subword_prefix:
        print("bpe with continuing subword prefix (##)")
        trainer = BpeTrainer(vocab_size=args.vocab_size, 
                            unk_token=unk_token, special_tokens=spl_tokens,
                            continuing_subword_prefix="##")
    else:
        print("bpe no continuing subword prefix (##)")
        trainer = BpeTrainer(vocab_size=args.vocab_size, 
                            unk_token=unk_token, special_tokens=spl_tokens)   

elif args.tokenizer == 'uni':
    tokenizer = Tokenizer(Unigram())
    trainer = UnigramTrainer(vocab_size=args.vocab_size, 
                        unk_token=unk_token, special_tokens=spl_tokens
                        )

elif args.tokenizer == 'wpc':
    tokenizer = Tokenizer(WordPiece(unk_token = unk_token))
    if args.continuing_subword_prefix:
        trainer = WordPieceTrainer(vocab_size=args.vocab_size,
                                    special_tokens = spl_tokens
                                    )    
    else:
        trainer = WordPieceTrainer(vocab_size=args.vocab_size, 
                            unk_token=unk_token, special_tokens=spl_tokens,
                            continuing_subword_prefix="")   


tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(input_files, trainer)
# normalizer = AutoTokenizer.from_pretrained("bert-base-uncased").backend_tokenizer.normalizer
# tokenizer.normalizer = normalizer
json_path = os.path.join(args.ext_vocab_output_dir, f'{args.tokenizer}_{args.vocab_size}.json')
# make sure the directory exists
os.makedirs(args.ext_vocab_output_dir, exist_ok=True)
tokenizer.save(json_path)

# output txt from json dict
ext_vocab_output_path = os.path.join(args.ext_vocab_output_dir, f'{args.tokenizer}_{args.vocab_size}.txt')
basic_vocab_output_path = os.path.join(args.ext_vocab_output_dir, 'basic_vocab.txt')

# learned vocab in json dict
tokenizer_file = json_path
with open(tokenizer_file, 'r', encoding='utf-8') as file:
    tokenizer_dic = json.load(file)

with open(ext_vocab_output_path, 'w', encoding='utf-8') as file, open(basic_vocab_output_path, 'w', encoding='utf-8') as file_c:
    if args.tokenizer == 'uni':
        for token, score in tokenizer_dic['model']['vocab']:
            if len(token) > 0:
                file.write(f"{token}\n")
                if (len(token) == 1) or (token in spl_tokens):
                    file_c.write(f"{token}\n")
    else:
        for token, index in tokenizer_dic['model']['vocab'].items():
            if len(token) > 0:
                file.write(f"{token}\n")
                if (len(token) == 1) or (token in spl_tokens):
                    file_c.write(f"{token}\n")