#pragma once

#include <cstdint>


namespace wrap_gemm {
void wrap_cuda_memcpy_2d_async(intptr_t dst_intptr, size_t dpitch, intptr_t src_intptr, size_t spitch, size_t width, size_t height, int cuda_memcpy_kind, intptr_t stream_intptr);

}