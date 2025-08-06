import math
from collections import defaultdict
from utils.vocab_builder import *
from utils.tokenizer import CharLevelTokenizer
from transformers import AutoConfig
from reader.word_reader import WordDataset
from torch.utils.data import DataLoader, SequentialSampler

CONT_PREFIX = None
CONT_PREFIX_ID = None

def update_scores(cnt_dict):
    total = sum(cnt_dict.values())
    score_dict = {}
    for key, cnt in cnt_dict.items():
        score_dict[key] = -math.log(cnt / total)
    return score_dict

def build_tree_ids_pair_with_freq(wiki_path, config_path, model_path, 
                         preload_vocab_path, pickle_output_path,
                         batch_size=32, save_steps=500000, skip_chunks=0):
    # choose device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    # prepare model
    from model.fast_r2d2_insideoutside_new import FastR2D2Plus
    config = AutoConfig.from_pretrained(config_path)
    model = FastR2D2Plus(config) # all_config integrate r2d2 and gpt 
    if os.path.exists(os.path.join(model_path, f'model.bin')):
        model.from_pretrain(os.path.join(model_path, f'model.bin'))
    else:
        print("WARNING: no model.bin in pretrain dir!")
    parser = model.parser
    parser.to(device)
    parser.eval()

    # prepare data
    char_tokenizer = CharLevelTokenizer()
    char_tokenizer.load_from_vocab_file(preload_vocab_path)
    dataset = WordDataset(wiki_path, char_tokenizer) 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=SequentialSampler(dataset),
                            collate_fn=dataset.collate_batch,
                            num_workers=1)
    epoch_iterator = tqdm(dataloader, desc="Iteration")

    # inference: create splits
    tree_ids_freq_pairs = []
    for step, inputs in enumerate(epoch_iterator):
        with torch.no_grad():
            inputs_device = {}
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs_device[k] = v.to(device)
            try:
                with torch.no_grad():
                    s_indices, _ = parser(**inputs_device)
            except Exception as e:
                print(inputs_device['input_ids'])

            s_indices = s_indices.cpu().data.numpy()
            ids_np = inputs['input_ids'].cpu().data.numpy()
            seq_lens = inputs['attention_mask'].sum(dim=-1).cpu().data.numpy()
            freq  = inputs['freqs']
            
            for sent_i, seq_len in enumerate(seq_lens):
                tree_ids_freq_pairs.append([s_indices[sent_i][:seq_len-1], ids_np[sent_i, :seq_len], freq[sent_i]])
        
    if len(tree_ids_freq_pairs) > 0:
        with open(''.join([pickle_output_path, '.pkl']), mode='wb') as fout:
            pickle.dump(tree_ids_freq_pairs, fout)

def count_basic_vocab(input_path, vocab_model):
    """
        count the freq of the basic vocab entries
    """
    with open(input_path, mode='rb') as f_in:
        ut_list = pickle.load(f_in)
   
    for indices, ids, freq in tqdm(ut_list, desc="count_basic_vocab"):
        if len(ids) == 0:
            continue
        elif len(ids) == 1:
            vocab_model.add(ids, cnt=freq)
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        to_visit = [root]
        while len(to_visit) > 0:
            top = to_visit.pop(-1)

            if top.left is not None and top.right is not None:
                to_visit.append(top.right)
                to_visit.append(top.left)
            else:
                if top.i == 0 or not CONT_PREFIX:
                    vocab_model.add(ids[top.i:top.j + 1], cnt=freq) # pass is_first = top.i==0
                else: ## continuing subword
                    word_ids = np.insert(ids[top.i:top.j + 1], 0, CONT_PREFIX_ID)
                    vocab_model.add(word_ids, cnt=freq)
        
    return vocab_model

def count_bigrams(input_path, vocab_model):
    with open(input_path, mode='rb') as f_in:
        ut_list = pickle.load(f_in)

    new_vocab_model = WordTree(len(vocab_model.basic_entries))

    def recursive_count_bigram(node, ids, freq):
        if node.left is not None and node.right is not None:
            left_hit = recursive_count_bigram(node.left, ids, freq)
            right_hit = recursive_count_bigram(node.right, ids, freq)
            if left_hit and right_hit:
                ## ready to merge
                ## check if self is hit
                if not CONT_PREFIX:
                    if vocab_model.has(ids[node.i : node.j + 1]):
                        return True
                    else:
                        new_vocab_model.add(ids[node.i : node.j + 1], cnt=freq)
                        return False
                else: ## support '##' mode
                    if node.i == 0:
                        word_ids = ids[node.i : node.j + 1]
                    else:
                        word_ids = np.insert(ids[node.i : node.j + 1], 0, CONT_PREFIX_ID)
                    if vocab_model.has(word_ids):
                        return True
                    else:
                        new_vocab_model.add(word_ids, cnt=freq)
                        return False
            else:
                return False
        else:
            return True

    for indices, ids, freq in tqdm(ut_list, desc="count_bigrams"):
        if len(ids) == 0:
            continue
        elif len(ids) == 1:
            if not vocab_model.has(ids):
                new_vocab_model.add(ids, cnt=freq)
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        recursive_count_bigram(root, ids, freq)

    return new_vocab_model

def build_heuristic_vocab(file_path, basic_vocab_size, threshold=5, output_path=None):
    from utils.vocab_builder import WordTree
    # save an id for continuing prefix
    if CONT_PREFIX:
        vocab_model = WordTree(basic_vocab_size+1)
    else:
        vocab_model = WordTree(basic_vocab_size)

    print(f'basic vocab size  {basic_vocab_size}')
    count_basic_vocab(file_path, vocab_model) 
    prev_size = vocab_model.size()

    while True:
        print(f'current size: {prev_size}')
        new_vocab_model = count_bigrams(file_path, vocab_model)
        new_vocab_model.truncate_by_count(threshold)

        vocab_model.merge(new_vocab_model)

        if vocab_model.size() == prev_size:
            break
        prev_size = vocab_model.size()

    # save heuristic vocab
    if output_path is not None:
        with codecs.open(output_path, mode='w') as f_out:
            for ids, count in vocab_model.vocab_iterator():
                print(f"{','.join([str(_) for _ in ids])}\t{count}", file=f_out)
    return vocab_model

def build_character_vocab(input_path, basic_vocab_size, expected_vocab_size, threshold, output_path, 
                            p_rate=0.2, continuing_subword_prefix=None):
    """
        <MAIN ENTRY POINT>
        expected_vocab_size: exclude all the basic vocab entries 
    """
    global CONT_PREFIX, CONT_PREFIX_ID
    CONT_PREFIX = continuing_subword_prefix
    print(f"continuing subword prefix is {continuing_subword_prefix}.")
    if CONT_PREFIX is not None:
        CONT_PREFIX_ID = basic_vocab_size

    word_tree = build_heuristic_vocab(input_path, basic_vocab_size=basic_vocab_size, threshold=5, 
                                      output_path='vocab_mining/heuristic_vocab') # entry, freq
    # convert wordtree to dict
    vocab_model = {}
    char_count = 0
    for ids, cnt in word_tree.vocab_iterator():
        key = to_ids_key(ids)
        vocab_model[key] = cnt
    with open(input_path, mode='rb') as f:
        ut_list = pickle.load(f)
    current_size = len(vocab_model)
    print(f'current size: {current_size}')
    while current_size > expected_vocab_size:
        # estimate token counts
        score_model = update_scores(vocab_model)
        vocab_model = unigram_E_step(ut_list, score_model)
        score_model = update_scores(vocab_model)
        # maxmize delta loss
        delta_losses = unigram_M_step(ut_list, score_model)
        sorted_items = sorted(delta_losses.items(), key=lambda x: x[1])
        to_remove = min(int(p_rate * len(sorted_items)), len(vocab_model) - expected_vocab_size)
        for key, _ in sorted_items[:to_remove]:
            del vocab_model[key]
        current_size = len(vocab_model)
        print(f'current_size: {current_size}')
    if output_path is not None:
        with codecs.open(output_path, mode='w') as f_out:
            for ids, count in sorted(vocab_model.items(), key=lambda x: -x[1]):
                f_out.write(f"{ids}\t{count}\n")

def unigram_E_step(pair_list, vocab_model):
    """
    Estimate the parameters of the current vocabulary: word frequency
    The method of counting word frequency: depends on the external parse tree (top-down hit)
    """
    count_dict = defaultdict(int)
    def best_segment(root, input_ids):
        if root.i == root.j:
            # if the current node is a leaf node, 
            # directly count the word frequency
            if CONT_PREFIX:
                if root.i == 0:
                    word_ids = input_ids[root.i: root.j + 1]
                else:
                    word_ids = np.insert(input_ids[root.i: root.j + 1], 0, CONT_PREFIX_ID)
            else:
                word_ids = input_ids[root.i: root.j + 1]
            key = to_ids_key(word_ids)
            # assert key in vocab_model
            score = vocab_model.get(key, float('inf'))
            return score, [word_ids]
        else:
            # if the current node is not a leaf node, 
            # recursively count the word frequency of the left and right subtrees
            left_score, left_tokens = best_segment(root.left, input_ids)
            right_score, right_tokens = best_segment(root.right, input_ids)
            if CONT_PREFIX:
                if root.i == 0:
                    word_ids = input_ids[root.i: root.j + 1]
                else:
                    word_ids = np.insert(input_ids[root.i: root.j + 1], 0, CONT_PREFIX_ID)
            else:
                word_ids = input_ids[root.i: root.j + 1]
            key = to_ids_key(word_ids)
            hit_score = vocab_model.get(key, float('inf'))
            
            if left_score + right_score > hit_score:
                return hit_score, [word_ids]

            return left_score + right_score, left_tokens + right_tokens

    for indices, ids, freq in tqdm(pair_list, desc="E step"):
        if len(ids) == 0:
            continue
        elif len(ids) == 1:
            key = to_ids_key(ids)
            count_dict[key] += freq
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        score, segments = best_segment(root, ids)

        for segment in segments:
            key = to_ids_key(segment)
            count_dict[key] += freq
        
    return count_dict

def unigram_M_step(pair_list, vocab_scores):
    """
        Calculate delta_loss to remove words with smallest delta_loss from current vocabulary,
        i.e. adjust parameters (word frequency) using max algorithm.
        delta_loss: left_H + right_H - parent_H (the cost of removing the word)
        :param pair_list: List[List[splits, ids, freq]]
        :param vocab_scores: WordTree
    """

    def delta_loss(root, input_ids, delta_loss_record):
        if root.i == root.j:
            if CONT_PREFIX:
                if root.i == 0:
                    word_ids = input_ids[root.i: root.j + 1]
                else:
                    word_ids = np.insert(input_ids[root.i: root.j + 1], 0, CONT_PREFIX_ID)
            else:
                word_ids = input_ids[root.i: root.j + 1]
            key = to_ids_key(word_ids)
            # assert key in vocab_scores
            score = vocab_scores.get(key, float('inf'))
            return score, [root]
        else:
            left_score, left_nodes = delta_loss(root.left, input_ids, delta_loss_record)
            right_score, right_nodes = delta_loss(root.right, input_ids, delta_loss_record)
            if CONT_PREFIX:
                if root.i == 0:
                    word_ids = input_ids[root.i: root.j + 1]
                else:
                    word_ids = np.insert(input_ids[root.i: root.j + 1], 0, CONT_PREFIX_ID)
            else:
                word_ids = input_ids[root.i: root.j + 1]
            key = to_ids_key(word_ids)
            hit_score = vocab_scores.get(key, float('inf'))
            delta_loss_record[root] += max(left_score + right_score - hit_score, 0)
            if left_score + right_score > hit_score:
                return hit_score, [root]

            return left_score + right_score, left_nodes + right_nodes

    d_loss_sum = defaultdict(int)
    for indices, ids, freq in tqdm(pair_list, desc="M step"):
        if len(ids) == 0:
            continue
        elif len(ids) == 1:
            # since we don't delete any basic vocab entries
            # no need to track the delta_loss of them
            continue
        
        root = get_tree_from_merge_trajectory(indices, len(ids))
        delta_loss_record = defaultdict(int)
        _, seg_nodes = delta_loss(root, ids, delta_loss_record)
        for node in seg_nodes:
            if node.i != node.j:
                d_loss = delta_loss_record[node]
                if CONT_PREFIX:
                    if node.i == 0:
                        word_ids = ids[node.i: node.j + 1]
                    else:
                        word_ids = np.insert(ids[node.i: node.j + 1], 0, CONT_PREFIX_ID)
                else:
                    word_ids = ids[node.i: node.j + 1]
                key = to_ids_key(word_ids)
                d_loss_sum[key] += d_loss * freq 
        
    return d_loss_sum

def make_vocab_readable(vocab_path, tokenizer, continuing_subword_prefix_id=None, output_path=None):
    """
    print(f"{','.join([str(_) for _ in ids])}\t{count}", file=f_out)
    """
    vocab_model = {}
    with codecs.open(vocab_path, mode='r', encoding='utf-8') as f_in:
        for l in f_in.readlines():
            key, cnt = l.strip().split('\t')
            vocab_model[key] = int(cnt)
    
    with codecs.open(f'{output_path}.readable', mode='w', encoding='utf-8') as f_out:
        for key, cnt in sorted(vocab_model.items(), key=lambda x: -x[1]):
            word_ids = [int(x) for x in key.split(",")]
            if word_ids[0] != continuing_subword_prefix_id:
                tokenized = tokenizer.convert_ids_to_tokens(word_ids)
            else:
                tokenized = [CONT_PREFIX]+tokenizer.convert_ids_to_tokens(word_ids[1:])
            print(f'{"".join(tokenized)}\t{cnt}', file=f_out)

if __name__ == '__main__':
    import os
    import sys, argparse
    cmd = argparse.ArgumentParser("Arguments for vocab mining")
    cmd.add_argument('--model_name', required=True, type=str)
    cmd.add_argument('--ckpt_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str)
    cmd.add_argument('--continuing_subword_prefix', type=str, default="##")
    cmd.add_argument('--vocab_size', type=int, default=30000)
    
    args = cmd.parse_args(sys.argv[1:])
    
    model_name = args.model_name # 'allobj_span_uncased1w'
    ckpt_dir = args.ckpt_dir # out
    r2d2_path = '.'
    data_path = args.data_path
    model_path = os.path.join(ckpt_dir, model_name)
    config_path = os.path.join(model_path, "all_config.json")
    preload_vocab_path = os.path.join(model_path, 'vocab.txt')
    output_base = os.path.join(r2d2_path, 'vocab_mining')
    output_dir = os.path.join(output_base, model_name)
    os.makedirs(output_dir, exist_ok=True)
    pickle_output_path = os.path.join(output_dir, model_name)
    vocab_output_path = os.path.join(output_dir, 'vocab_mining.out')
    tree_ids_path = os.path.join(output_dir, model_name+'.pkl')
    vocab_path = vocab_output_path
    output_path = os.path.join(output_dir, 'vocab_mining_read.out')
    
    char_tokenizer = CharLevelTokenizer()
    char_tokenizer.load_from_vocab_file(preload_vocab_path)
    
    build_tree_ids_pair_with_freq(data_path, config_path, model_path, 
                         preload_vocab_path, pickle_output_path, 
                         batch_size=1024, save_steps=500000, skip_chunks=0)
    basic_vocab_size = len(char_tokenizer.idx2vocab)
    print(f"basic_vocab_size: {basic_vocab_size}")
    build_character_vocab(input_path=tree_ids_path, basic_vocab_size=basic_vocab_size, expected_vocab_size=args.vocab_size, threshold=5, output_path=vocab_output_path, 
                            p_rate=0.2, continuing_subword_prefix=args.continuing_subword_prefix)
    make_vocab_readable(vocab_path=vocab_output_path, tokenizer=char_tokenizer,
                        continuing_subword_prefix_id=basic_vocab_size,
                        output_path=output_path)