// Host tensor data is not required to be page-aligned. No model files needed.
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (!dev) { std::fprintf(stderr, "SKIP: MTL0 unavailable\n"); return 77; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { return 1; }
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); ggml_backend_free(backend); return 1; }
    const size_t size = ggml_nbytes(tensor);
    std::vector<unsigned char> source(size + 1), target(size + 1, 0);
    for (size_t i = 0; i < size; ++i) { source[i + 1] = (unsigned char)(i % 251); }
    ggml_backend_tensor_set(tensor, source.data() + 1, 0, size);
    ggml_backend_tensor_get(tensor, target.data() + 1, 0, size);
    bool equal = std::memcmp(source.data() + 1, target.data() + 1, size) == 0;
    // Exercise an unaligned subrange as well as the whole tensor.
    ggml_backend_tensor_set(tensor, source.data() + 1, 4, size - 4);
    ggml_backend_tensor_get(tensor, target.data() + 1, 4, size - 4);
    equal = equal && std::memcmp(source.data() + 1, target.data() + 1, size - 4) == 0;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, "unaligned Metal transfer: %s\n", equal ? "PASS" : "FAIL");
    return equal ? 0 : 1;
}
