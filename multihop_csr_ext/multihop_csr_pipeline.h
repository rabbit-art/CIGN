#pragma once

#include <torch/extension.h>
#include <memory>
#include <string>
#include <vector>

class MultiHopCsrPlan {
public:
    MultiHopCsrPlan(
        torch::Tensor p_crow,
        torch::Tensor p_col,
        torch::Tensor p_values,
        torch::Tensor pt_crow,
        torch::Tensor pt_col,
        torch::Tensor pt_values,
        int64_t rows,
        int64_t dense_cols,
        int64_t algorithm);

    ~MultiHopCsrPlan();

    void prepare(torch::Tensor x);
    std::vector<torch::Tensor> forward3(torch::Tensor x);
    torch::Tensor backward3(
        torch::Tensor grad_y1,
        torch::Tensor grad_y2,
        torch::Tensor grad_y3);

    int64_t algorithm() const;
    int64_t workspace_bytes() const;
    std::string info() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
