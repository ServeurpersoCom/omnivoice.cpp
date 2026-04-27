#pragma once
// debug.h : tensor dump and compare helpers for Python vs GGML validation.
// Dumps raw f32 arrays to binary files, both backends convert to f32 before
// dump.
// File format : [int32 ndims] [int32 dim0] [int32 dim1] ... [float data...]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct DebugDumper {
    char dir[512];
    bool enabled;
};

static void debug_init(DebugDumper * d, const char * dir) {
    d->enabled = (dir != nullptr);
    if (d->enabled) {
        snprintf(d->dir, sizeof(d->dir), "%s", dir);
    }
}

// Dump f32 tensor to binary file.
// Format : [ndims:i32] [shape:i32 x ndims] [data:f32 x numel]
static void debug_dump(const DebugDumper * d, const char * name, const float * data, const int * shape, int ndims) {
    if (!d->enabled) {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", d->dir, name);

    int numel = 1;
    for (int i = 0; i < ndims; i++) {
        numel *= shape[i];
    }
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Debug] cannot write %s\n", path);
        return;
    }
    fwrite(&ndims, sizeof(int32_t), 1, f);
    fwrite(shape, sizeof(int32_t), ndims, f);
    fwrite(data, sizeof(float), numel, f);
    fclose(f);

    // First 4 values for quick sanity check on stderr.
    fprintf(stderr, "[Debug] %s: [", name);
    for (int i = 0; i < ndims; i++) {
        fprintf(stderr, "%s%d", i ? ", " : "", shape[i]);
    }
    fprintf(stderr, "] first4:");
    for (int i = 0; i < 4 && i < numel; i++) {
        fprintf(stderr, " %.6f", data[i]);
    }
    fprintf(stderr, "\n");
}

// Convenience : dump 1D tensor [n].
static void debug_dump_1d(const DebugDumper * d, const char * name, const float * data, int n) {
    debug_dump(d, name, data, &n, 1);
}

// Convenience : dump 2D tensor [rows, cols].
static void debug_dump_2d(const DebugDumper * d, const char * name, const float * data, int dim0, int dim1) {
    int shape[2] = { dim0, dim1 };
    debug_dump(d, name, data, shape, 2);
}

// Convenience : dump 3D tensor [d0, d1, d2].
static void debug_dump_3d(const DebugDumper * d, const char * name, const float * data, int d0, int d1, int d2) {
    int shape[3] = { d0, d1, d2 };
    debug_dump(d, name, data, shape, 3);
}

// Convenience : dump 4D tensor [d0, d1, d2, d3].
static void
debug_dump_4d(const DebugDumper * d, const char * name, const float * data, int d0, int d1, int d2, int d3) {
    int shape[4] = { d0, d1, d2, d3 };
    debug_dump(d, name, data, shape, 4);
}

// Cast a stream of int32 values to f32 in place into a temporary buffer and
// dump under the given name. Token comparisons are then expressed as cossim
// over float values, with exact match recoverable via integer compare on the
// loader side.
static void debug_dump_i32_as_f32(const DebugDumper * d,
                                  const char *        name,
                                  const int32_t *     data,
                                  const int *         shape,
                                  int                 ndims) {
    if (!d->enabled) {
        return;
    }
    int numel = 1;
    for (int i = 0; i < ndims; i++) {
        numel *= shape[i];
    }
    std::vector<float> buf((size_t) numel);
    for (int i = 0; i < numel; i++) {
        buf[i] = (float) data[i];
    }
    debug_dump(d, name, buf.data(), shape, ndims);
}

// Cosine similarity between two f32 arrays.
static double debug_cosine_sim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na += (double) a[i] * (double) a[i];
        nb += (double) b[i] * (double) b[i];
    }
    if (na < 1e-30 || nb < 1e-30) {
        return 0.0;
    }
    return dot / (sqrt(na) * sqrt(nb));
}
