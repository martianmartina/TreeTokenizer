#include <torch/torch.h>
#include "py_backend.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::string name = std::string("TableManager");
    py::class_<TableManager>(m, name.c_str())
        .def(py::init([](const py::array_t<int>& seq_lens, const py::array_t<int>& group_ids,
                         const py::array_t<int>& merge_orders, 
                         size_t window_size, size_t cache_id_offset, size_t detach_cache_id_offset,
                         vector<py::array_t<int>>& span_ids)
                      { return new TableManager(seq_lens, group_ids, merge_orders, window_size, 
                                                cache_id_offset, detach_cache_id_offset, span_ids); }))
        .def("step", &TableManager::step)
        .def("best_trees", &TableManager::best_trees)
        .def("root_ids", &TableManager::root_ids)
        .def("prepare_bilm", &TableManager::prepare_bilm)
        .def("get_cached_bij", &TableManager::get_cached_bij)
        .def("batch_size", &TableManager::batch_size)
        .def("prepare_span_prediction", &TableManager::prepare_span_prediction)
        .def("is_finished", &TableManager::is_finished);
    name = std::string("SpanTokenizer");
    py::class_<SpanTokenizer>(m, name.c_str())
        .def(py::init([](vector<py::array_t<int>>& dictionary, int max_entry_id) {
            return new SpanTokenizer(dictionary, max_entry_id);
        }))
        .def("tokenize", &SpanTokenizer::tokenize);
}