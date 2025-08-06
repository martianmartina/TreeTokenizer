// coding=utf-8
// Copyright (c) 2022 Ant Group
// Author: Xiang Hu

#pragma once
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
using namespace std;

struct TableCell;

struct TreeNode
{
    int cache_id;
    TreeNode *left;
    TreeNode *right;
    TableCell *owner;
    float log_p;
};

struct TableCell
{
    TreeNode *beams;
    uint * splits;           //record valid split points
    float *candidates_log_p; //keep the log_p of all possible combinations
    bool eliminated;         //whether the cell is pruned from the table.
    bool is_term;            //whether the cell is terminal cell
    float max_log_p;
    float max_left_log_p_;
    float max_right_log_p_;
    uint beam_size;
    uint best_tree_idx;
    uint table_id;
    uint i;
    uint j;
    int cell_idx;
};

struct CellRange
{
    int start;    // the first cell to update in the layer_i th row of m_pActiveCells
    int end;      // cells in layer_i: [start, end] are updated, end included.
    int term_len; // the length of all terminal nodes;
    // uint cache_id_offset;
    uint seq_len;
    uint layer_i; // the i_th layer of the m_pActiveCells
    uint batch_id_offset; // the total number of assigned ids for the current pyramid
};

struct ExportNode
{
    int cache_id;
    int left_i;
    int right_i;
    int left_j;
    int right_j;
    int left_idx;
    int right_idx;
    float log_p;
};

struct ExportCell
{
    int best_tree_idx;
    bool detach;
    vector<ExportNode> nodes;
};

class TablesManager
{
private:
    bool m_bEncoding;
    bool m_bDirectional;
    bool m_bHasTrajectories;
    bool m_bBigramOnly;

    //CUDA pointers;
    TableCell ***m_pActiveCells; //batch of active cells for pruning [batch_size * window_size * seq_len_i]
    TableCell **m_pCells;        //batch of all cells. [batch_size * seq_len_i * seq_len_i]
    // uint *m_pSeqlens;            //batch of seq_lens.
    uint *m_pCellOffsets; // Let c_arr denote the array of all cells to update, cell offsets denote the start index of each batch.
    uint **m_pLeftmost;   // for position i, record the longest non empty span j, where j >= i
    uint **m_pRightmost;
    CellRange *m_pUpdateRange; //batch of next cells to update
    // for cuda memory dealloc
    float *m_pLogProbCache;
    uint m_iLogProbOffset;
    TreeNode *m_pNodeCache;
    uint * m_pSplitCache;
    uint m_iNodeOffset;
    uint m_iCellIdxOffset;

    at::Tensor m_tSeqlens;
    at::Tensor m_tMergePos;

    uint m_iTotalLen;
    uint m_iGroupSize;
    uint m_iCurrentStep;
    uint m_iBeamSize;
    uint m_iWindowSize;
    uint m_iMaxSeqlen;
    uint m_iCacheOffset;
    uint m_iDetachedCacheOffset;
    uint m_iEmptyCacheId;
    // std::vector<uint> m_vSeqlens;

    std::vector<uint> m_vBeamSizes;

public:
    TablesManager(bool directional, uint window_size, uint beam_size);
    ~TablesManager();
    void encoding_start(const at::Tensor & seq_lens, uint cache_id_offset, uint detached_id_offset, uint empty_cache_id); // return the size of tensor buffer.
    void set_merge_trajectories(const at::Tensor & indices);
    void set_bigram_only();
    vector<uint> step(const at::Tensor & cache_ids, const at::Tensor & log_p_ids, const at::Tensor & span_lens, 
                      const at::Tensor & bigram_score_cache, const at::Tensor & noise); //return total_batch_num, total_node_num
    void prepare_bilm(const at::Tensor & cache_ids, uint bos, uint eos);
    void prepare_span_bilm(const at::Tensor & cache_ids, const at::Tensor & spans_batch, uint bos, uint eos);
    void beam_select(const at::Tensor & indices); // Specify top k combinations for latest updated cells.
    void step_over(const at::Tensor & beam_log_p, const at::Tensor & candidates_log_p);
    void step_over_trivial(); //simple version of step over
    uint current_step();
    void encoding_over();
    void gather_root_ids(const at::Tensor & root_ids);
    void recover_sampled_trees(const at::Tensor & span_masks, const at::Tensor & targets, const at::Tensor & split_points);
    void recover_split_cache_ids(const at::Tensor & split_points, const at::Tensor & split_indices, const at::Tensor & split_cache_ids);
    bool finished();
    int total_len();
    size_t batch_size();
    vector<vector<ExportCell>> dump_cells();
};

void gather2d(const at::Tensor & src, const at::Tensor & index, const at::Tensor & tgt, const at::Tensor & id2batch);