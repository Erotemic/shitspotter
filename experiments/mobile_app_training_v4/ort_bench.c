/* ort_bench.c — minimal ONNX Runtime latency benchmark for Android ARM64.
 *
 * Compiles with Android NDK against libonnxruntime.so from the ORT 1.19 AAR.
 * Handles any model including multi-input ones (e.g. DEIMv2 images +
 * orig_target_sizes).
 *
 * Usage:
 *   ort_bench <model.onnx> [iters=50] [warmup=5] [ep=cpu|nnapi]
 *
 * Output (one line to stdout):
 *   mean_ms=<N> fps=<N> iters=<N> ep=<ep> model=<path>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"

static const OrtApi* g_ort = NULL;

#define ORT_CHECK(expr) do { \
    OrtStatus* _s = (expr); \
    if (_s) { \
        fprintf(stderr, "ORT error at %s:%d: %s\n", \
                __FILE__, __LINE__, g_ort->GetErrorMessage(_s)); \
        g_ort->ReleaseStatus(_s); \
        exit(1); \
    } \
} while (0)

static double mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: ort_bench <model.onnx> [iters=50] [warmup=5] [ep=cpu|nnapi]\n");
        return 1;
    }
    const char* model_path = argv[1];
    int iters  = argc > 2 ? atoi(argv[2]) : 50;
    int warmup = argc > 3 ? atoi(argv[3]) : 5;
    const char* ep = argc > 4 ? argv[4] : "cpu";

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) { fprintf(stderr, "OrtGetApiBase failed\n"); return 1; }

    OrtEnv* env = NULL;
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));

    OrtSessionOptions* opts = NULL;
    ORT_CHECK(g_ort->CreateSessionOptions(&opts));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(opts, 1));
    ORT_CHECK(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    if (strcmp(ep, "nnapi") == 0) {
        uint32_t flags = 0;
        OrtStatus* s = OrtSessionOptionsAppendExecutionProvider_Nnapi(opts, flags);
        if (s) {
            fprintf(stderr, "NNAPI unavailable, falling back to CPU: %s\n",
                    g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
            ep = "cpu";
        }
    }

    OrtSession* session = NULL;
    ORT_CHECK(g_ort->CreateSession(env, model_path, opts, &session));

    OrtAllocator* allocator = NULL;
    ORT_CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    OrtMemoryInfo* mem_info = NULL;
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                          &mem_info));

    /* ---- build input tensors ------------------------------------------ */
    size_t n_in = 0;
    ORT_CHECK(g_ort->SessionGetInputCount(session, &n_in));

    const char** in_names = (const char**)calloc(n_in, sizeof(char*));
    OrtValue**   in_vals  = (OrtValue**)calloc(n_in, sizeof(OrtValue*));

    /* First pass: find float32 image input to extract H/W for int64 inputs */
    int64_t img_h = 320, img_w = 320;

    for (size_t i = 0; i < n_in; i++) {
        char* name = NULL;
        ORT_CHECK(g_ort->SessionGetInputName(session, i, allocator, &name));
        in_names[i] = name;

        OrtTypeInfo* ti = NULL;
        ORT_CHECK(g_ort->SessionGetInputTypeInfo(session, i, &ti));

        const OrtTensorTypeAndShapeInfo* tsi = NULL;
        ORT_CHECK(g_ort->CastTypeInfoToTensorInfo(ti, &tsi));

        ONNXTensorElementDataType dtype;
        ORT_CHECK(g_ort->GetTensorElementType(tsi, &dtype));

        size_t ndim = 0;
        ORT_CHECK(g_ort->GetDimensionsCount(tsi, &ndim));

        int64_t* dims = (int64_t*)malloc(ndim * sizeof(int64_t));
        ORT_CHECK(g_ort->GetDimensions(tsi, dims, ndim));

        /* Clamp dynamic dims to 1; keep static dims */
        for (size_t d = 0; d < ndim; d++)
            if (dims[d] < 1) dims[d] = 1;

        if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && ndim == 4) {
            img_h = dims[2];
            img_w = dims[3];
        }

        free(dims);
        g_ort->ReleaseTypeInfo(ti);
    }

    /* Second pass: allocate tensors */
    for (size_t i = 0; i < n_in; i++) {
        OrtTypeInfo* ti = NULL;
        ORT_CHECK(g_ort->SessionGetInputTypeInfo(session, i, &ti));

        const OrtTensorTypeAndShapeInfo* tsi = NULL;
        ORT_CHECK(g_ort->CastTypeInfoToTensorInfo(ti, &tsi));

        ONNXTensorElementDataType dtype;
        ORT_CHECK(g_ort->GetTensorElementType(tsi, &dtype));

        size_t ndim = 0;
        ORT_CHECK(g_ort->GetDimensionsCount(tsi, &ndim));

        int64_t* dims = (int64_t*)malloc(ndim * sizeof(int64_t));
        ORT_CHECK(g_ort->GetDimensions(tsi, dims, ndim));

        for (size_t d = 0; d < ndim; d++)
            if (dims[d] < 1) dims[d] = 1;

        size_t n_elems = 1;
        for (size_t d = 0; d < ndim; d++) n_elems *= (size_t)dims[d];

        size_t elem_sz = (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ? 8 : 4;
        void* buf = calloc(n_elems, elem_sz);

        if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && n_elems == 2) {
            /* DEIMv2 orig_target_sizes — fill with [H, W] */
            int64_t* iv = (int64_t*)buf;
            iv[0] = img_h;
            iv[1] = img_w;
        } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            /* Gray-ish image: 0.5 / 255 normalised */
            float* fv = (float*)buf;
            for (size_t j = 0; j < n_elems; j++) fv[j] = 0.5f / 255.0f;
        }

        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, buf, n_elems * elem_sz, dims, ndim, dtype, &in_vals[i]));

        free(dims);
        g_ort->ReleaseTypeInfo(ti);
    }

    /* ---- output names -------------------------------------------------- */
    size_t n_out = 0;
    ORT_CHECK(g_ort->SessionGetOutputCount(session, &n_out));

    const char** out_names = (const char**)calloc(n_out, sizeof(char*));
    OrtValue**   out_vals  = (OrtValue**)calloc(n_out, sizeof(OrtValue*));

    for (size_t i = 0; i < n_out; i++) {
        char* name = NULL;
        ORT_CHECK(g_ort->SessionGetOutputName(session, i, allocator, &name));
        out_names[i] = name;
    }

    OrtRunOptions* run_opts = NULL;
    ORT_CHECK(g_ort->CreateRunOptions(&run_opts));

    /* ---- warmup -------------------------------------------------------- */
    for (int i = 0; i < warmup; i++) {
        for (size_t j = 0; j < n_out; j++) {
            if (out_vals[j]) { g_ort->ReleaseValue(out_vals[j]); out_vals[j] = NULL; }
        }
        ORT_CHECK(g_ort->Run(session, run_opts,
            in_names, (const OrtValue* const*)in_vals, n_in,
            out_names, n_out, out_vals));
    }

    /* ---- benchmark ----------------------------------------------------- */
    double total_ms = 0.0;
    for (int i = 0; i < iters; i++) {
        for (size_t j = 0; j < n_out; j++) {
            if (out_vals[j]) { g_ort->ReleaseValue(out_vals[j]); out_vals[j] = NULL; }
        }
        double t0 = mono_ms();
        ORT_CHECK(g_ort->Run(session, run_opts,
            in_names, (const OrtValue* const*)in_vals, n_in,
            out_names, n_out, out_vals));
        total_ms += mono_ms() - t0;
    }

    double mean_ms = total_ms / iters;
    double fps = 1000.0 / mean_ms;
    printf("mean_ms=%.2f fps=%.2f iters=%d ep=%s model=%s\n",
           mean_ms, fps, iters, ep, model_path);

    return 0;
}
