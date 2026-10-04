#include "multihop_csr_pipeline.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define CHECK_CUSPARSE(EXPR)                                                     \
    do {                                                                         \
        cusparseStatus_t _status = (EXPR);                                       \
        TORCH_CHECK(                                                             \
            _status == CUSPARSE_STATUS_SUCCESS,                                  \
            "cuSPARSE call failed at ", __FILE__, ":", __LINE__,             \
            " status=", static_cast<int>(_status));                             \
    } while (0)

#define CHECK_CUDA(EXPR)                                                         \
    do {                                                                         \
        cudaError_t _status = (EXPR);                                            \
        TORCH_CHECK(                                                             \
            _status == cudaSuccess,                                              \
            "CUDA call failed at ", __FILE__, ":", __LINE__,                 \
            " error=", cudaGetErrorString(_status));                            \
    } while (0)

namespace {

cusparseSpMMAlg_t parse_algorithm(int64_t algorithm) {
    switch (algorithm) {
        case 1: return CUSPARSE_SPMM_CSR_ALG1;
        case 2: return CUSPARSE_SPMM_CSR_ALG2;
        case 3: return CUSPARSE_SPMM_CSR_ALG3;
        default:
            TORCH_CHECK(false, "MultiHopCsrPlan algorithm must be 1, 2, or 3; got ", algorithm);
    }
}

const char* algorithm_name(int64_t algorithm) {
    switch (algorithm) {
        case 1: return "CUSPARSE_SPMM_CSR_ALG1";
        case 2: return "CUSPARSE_SPMM_CSR_ALG2";
        case 3: return "CUSPARSE_SPMM_CSR_ALG3";
        default: return "UNKNOWN";
    }
}

void check_csr_tensor(const torch::Tensor& x, int device_index, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA.");
    TORCH_CHECK(x.get_device() == device_index, name, " is on wrong CUDA device.");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous.");
}

void check_dense(const torch::Tensor& x, int device_index, int64_t rows, int64_t cols, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA.");
    TORCH_CHECK(x.get_device() == device_index, name, " is on wrong CUDA device.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be float32.");
    TORCH_CHECK(x.dim() == 2, name, " must be 2-D.");
    TORCH_CHECK(x.size(0) == rows && x.size(1) == cols,
                name, " shape mismatch, expected [", rows, ", ", cols,
                "], got [", x.size(0), ", ", x.size(1), "].");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous row-major.");
}

__global__ void add_inplace_float4_kernel(float* __restrict__ dst,
                                          const float* __restrict__ src,
                                          int64_t n4) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 a = reinterpret_cast<float4*>(dst)[idx];
        float4 b = reinterpret_cast<const float4*>(src)[idx];
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        a.w += b.w;
        reinterpret_cast<float4*>(dst)[idx] = a;
    }
}

__global__ void add_inplace_scalar_kernel(float* __restrict__ dst,
                                          const float* __restrict__ src,
                                          int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}

void add_inplace(torch::Tensor dst, const torch::Tensor& src, cudaStream_t stream) {
    TORCH_CHECK(dst.numel() == src.numel(), "add_inplace numel mismatch.");
    int64_t n = dst.numel();
    constexpr int threads = 256;
    if ((n % 4) == 0 && (reinterpret_cast<uintptr_t>(dst.data_ptr<float>()) % 16) == 0 &&
        (reinterpret_cast<uintptr_t>(src.data_ptr<float>()) % 16) == 0) {
        int64_t n4 = n / 4;
        int blocks = static_cast<int>((n4 + threads - 1) / threads);
        add_inplace_float4_kernel<<<blocks, threads, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), n4);
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        add_inplace_scalar_kernel<<<blocks, threads, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), n);
    }
    CHECK_CUDA(cudaGetLastError());
}

}  // namespace

struct MultiHopCsrPlan::Impl {
    torch::Tensor p_crow;
    torch::Tensor p_col;
    torch::Tensor p_values;
    torch::Tensor pt_crow;
    torch::Tensor pt_col;
    torch::Tensor pt_values;
    torch::Tensor workspace;

    int64_t rows = 0;
    int64_t dense_cols = 0;
    int64_t nnz = 0;
    int64_t algorithm_id = 2;
    int device_index = -1;
    size_t workspace_size = 0;
    bool prepared = false;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matP = nullptr;
    cusparseSpMatDescr_t matPt = nullptr;
    cusparseDnMatDescr_t matB = nullptr;
    cusparseDnMatDescr_t matC = nullptr;
    cusparseSpMMAlg_t algorithm;

    Impl(torch::Tensor p_crow_,
         torch::Tensor p_col_,
         torch::Tensor p_values_,
         torch::Tensor pt_crow_,
         torch::Tensor pt_col_,
         torch::Tensor pt_values_,
         int64_t rows_,
         int64_t dense_cols_,
         int64_t algorithm_id_)
        : p_crow(std::move(p_crow_)),
          p_col(std::move(p_col_)),
          p_values(std::move(p_values_)),
          pt_crow(std::move(pt_crow_)),
          pt_col(std::move(pt_col_)),
          pt_values(std::move(pt_values_)),
          rows(rows_),
          dense_cols(dense_cols_),
          nnz(p_values.numel()),
          algorithm_id(algorithm_id_),
          algorithm(parse_algorithm(algorithm_id_)) {

        TORCH_CHECK(rows > 0 && dense_cols > 0, "rows and dense_cols must be positive.");
        TORCH_CHECK(p_values.is_cuda(), "CSR tensors must be CUDA.");
        device_index = p_values.get_device();

        check_csr_tensor(p_crow, device_index, "p_crow");
        check_csr_tensor(p_col, device_index, "p_col");
        check_csr_tensor(p_values, device_index, "p_values");
        check_csr_tensor(pt_crow, device_index, "pt_crow");
        check_csr_tensor(pt_col, device_index, "pt_col");
        check_csr_tensor(pt_values, device_index, "pt_values");

        TORCH_CHECK(p_crow.scalar_type() == torch::kInt32 &&
                    p_col.scalar_type() == torch::kInt32 &&
                    pt_crow.scalar_type() == torch::kInt32 &&
                    pt_col.scalar_type() == torch::kInt32,
                    "MultiHopCsrPlan requires int32 CSR indices.");
        TORCH_CHECK(p_values.scalar_type() == torch::kFloat32 &&
                    pt_values.scalar_type() == torch::kFloat32,
                    "MultiHopCsrPlan requires float32 CSR values.");
        TORCH_CHECK(p_crow.numel() == rows + 1 && pt_crow.numel() == rows + 1,
                    "CSR row-pointer length mismatch.");
        TORCH_CHECK(p_col.numel() == p_values.numel(), "P col/value nnz mismatch.");
        TORCH_CHECK(pt_col.numel() == pt_values.numel(), "P^T col/value nnz mismatch.");

        c10::cuda::CUDAGuard guard(device_index);
        CHECK_CUSPARSE(cusparseCreate(&handle));
        CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

        CHECK_CUSPARSE(cusparseCreateCsr(
            &matP, rows, rows, p_values.numel(),
            p_crow.data_ptr<int32_t>(), p_col.data_ptr<int32_t>(), p_values.data_ptr<float>(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matPt, rows, rows, pt_values.numel(),
            pt_crow.data_ptr<int32_t>(), pt_col.data_ptr<int32_t>(), pt_values.data_ptr<float>(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~Impl() {
        if (matB) cusparseDestroyDnMat(matB);
        if (matC) cusparseDestroyDnMat(matC);
        if (matP) cusparseDestroySpMat(matP);
        if (matPt) cusparseDestroySpMat(matPt);
        if (handle) cusparseDestroy(handle);
    }

    void set_stream() {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index);
        CHECK_CUSPARSE(cusparseSetStream(handle, stream));
    }

    void update_dense(torch::Tensor x, torch::Tensor out) {
        if (!matB) {
            CHECK_CUSPARSE(cusparseCreateDnMat(
                &matB, rows, dense_cols, dense_cols, x.data_ptr<float>(),
                CUDA_R_32F, CUSPARSE_ORDER_ROW));
        } else {
            CHECK_CUSPARSE(cusparseDnMatSetValues(matB, x.data_ptr<float>()));
        }
        if (!matC) {
            CHECK_CUSPARSE(cusparseCreateDnMat(
                &matC, rows, dense_cols, dense_cols, out.data_ptr<float>(),
                CUDA_R_32F, CUSPARSE_ORDER_ROW));
        } else {
            CHECK_CUSPARSE(cusparseDnMatSetValues(matC, out.data_ptr<float>()));
        }
    }

    size_t query_workspace(cusparseSpMatDescr_t A, torch::Tensor x, torch::Tensor out) {
        update_dense(x, out);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        size_t bytes = 0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            A,
            matB,
            &beta,
            matC,
            CUDA_R_32F,
            algorithm,
            &bytes));
        return bytes;
    }

    void prepare(torch::Tensor x) {
        check_dense(x, device_index, rows, dense_cols, "prepare input");
        c10::cuda::CUDAGuard guard(device_index);
        set_stream();
        auto out = torch::empty({rows, dense_cols}, x.options());
        size_t p_bytes = query_workspace(matP, x, out);
        size_t pt_bytes = query_workspace(matPt, x, out);
        workspace_size = std::max(p_bytes, pt_bytes);
        if (workspace_size > 0) {
            workspace = torch::empty(
                {static_cast<int64_t>(workspace_size)},
                torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
        } else {
            workspace = torch::Tensor();
        }
        prepared = true;
    }

    void spmm(cusparseSpMatDescr_t A, torch::Tensor x, torch::Tensor out) {
        TORCH_CHECK(prepared, "MultiHopCsrPlan.prepare() must be called before timed execution.");
        check_dense(x, device_index, rows, dense_cols, "SpMM input");
        check_dense(out, device_index, rows, dense_cols, "SpMM output");
        set_stream();
        update_dense(x, out);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        void* ws = workspace_size > 0 ? workspace.data_ptr() : nullptr;
        CHECK_CUSPARSE(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            A,
            matB,
            &beta,
            matC,
            CUDA_R_32F,
            algorithm,
            ws));
    }

    std::vector<torch::Tensor> forward3(torch::Tensor x) {
        check_dense(x, device_index, rows, dense_cols, "forward3 x");
        c10::cuda::CUDAGuard guard(device_index);
        auto y1 = torch::empty({rows, dense_cols}, x.options());
        auto y2 = torch::empty({rows, dense_cols}, x.options());
        auto y3 = torch::empty({rows, dense_cols}, x.options());
        spmm(matP, x, y1);
        spmm(matP, y1, y2);
        spmm(matP, y2, y3);
        return {y1, y2, y3};
    }

    torch::Tensor backward3(torch::Tensor g1, torch::Tensor g2, torch::Tensor g3) {
        check_dense(g1, device_index, rows, dense_cols, "backward3 g1");
        check_dense(g2, device_index, rows, dense_cols, "backward3 g2");
        check_dense(g3, device_index, rows, dense_cols, "backward3 g3");
        c10::cuda::CUDAGuard guard(device_index);
        set_stream();
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index);

        // Chain rule for y1=P x, y2=P y1, y3=P y2:
        // t2 = g2 + P^T g3
        // t1 = g1 + P^T t2
        // gx = P^T t1
        auto t2 = torch::empty({rows, dense_cols}, g1.options());
        auto t1 = torch::empty({rows, dense_cols}, g1.options());
        auto gx = torch::empty({rows, dense_cols}, g1.options());

        spmm(matPt, g3, t2);
        add_inplace(t2, g2, stream);
        spmm(matPt, t2, t1);
        add_inplace(t1, g1, stream);
        spmm(matPt, t1, gx);
        return gx;
    }
};

MultiHopCsrPlan::MultiHopCsrPlan(
    torch::Tensor p_crow,
    torch::Tensor p_col,
    torch::Tensor p_values,
    torch::Tensor pt_crow,
    torch::Tensor pt_col,
    torch::Tensor pt_values,
    int64_t rows,
    int64_t dense_cols,
    int64_t algorithm)
    : impl_(std::make_shared<Impl>(
          std::move(p_crow), std::move(p_col), std::move(p_values),
          std::move(pt_crow), std::move(pt_col), std::move(pt_values),
          rows, dense_cols, algorithm)) {}

MultiHopCsrPlan::~MultiHopCsrPlan() = default;

void MultiHopCsrPlan::prepare(torch::Tensor x) {
    impl_->prepare(std::move(x));
}

std::vector<torch::Tensor> MultiHopCsrPlan::forward3(torch::Tensor x) {
    return impl_->forward3(std::move(x));
}

torch::Tensor MultiHopCsrPlan::backward3(
    torch::Tensor grad_y1,
    torch::Tensor grad_y2,
    torch::Tensor grad_y3) {
    return impl_->backward3(
        std::move(grad_y1), std::move(grad_y2), std::move(grad_y3));
}

int64_t MultiHopCsrPlan::algorithm() const { return impl_->algorithm_id; }
int64_t MultiHopCsrPlan::workspace_bytes() const { return static_cast<int64_t>(impl_->workspace_size); }

std::string MultiHopCsrPlan::info() const {
    std::ostringstream oss;
    oss << "A3-v5 MultiHopCSR3, algorithm=" << algorithm_name(impl_->algorithm_id)
        << ", shape=[" << impl_->rows << "," << impl_->rows << "]"
        << ", nnz=" << impl_->nnz
        << ", dense_cols=" << impl_->dense_cols
        << ", workspace=" << impl_->workspace_size << " bytes"
        << ", device=cuda:" << impl_->device_index;
    return oss.str();
}
