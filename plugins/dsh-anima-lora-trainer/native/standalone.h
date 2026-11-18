#ifndef DSH_ANIMA_STANDALONE_H
#define DSH_ANIMA_STANDALONE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DshAnimaDtype {
    DSH_ANIMA_BF16 = 1,
    DSH_ANIMA_F32 = 2,
    DSH_ANIMA_F16 = 3,
} DshAnimaDtype;

typedef struct DshAnimaTensor {
    const char* name;
    const uint8_t* data;
    size_t bytes;
    const int64_t* shape;
    size_t dimensions;
    uint32_t dtype;
} DshAnimaTensor;

typedef struct DshAnimaSample {
    // RGB data, if given, has exactly height * width * 3 packed uint8 values.
    // The four token arrays have 512 int64 values each.
    const uint8_t* rgb;
    int64_t height;
    int64_t width;
    const int64_t* qwen_ids;
    const int64_t* qwen_mask;
    const int64_t* t5_ids;
    const int64_t* t5_mask;
    // With a null RGB pointer, the sample is already cached as these tensors.
    DshAnimaTensor latent;
    DshAnimaTensor context;
} DshAnimaSample;

typedef struct DshAnimaConfig {
    const DshAnimaTensor* weights;
    size_t weight_count;
    size_t sample_count;
    const char* frontend_dir;
    int64_t steps;
    int64_t rank;
    int64_t seed;
    int64_t log_every;
    double lr;
} DshAnimaConfig;

typedef struct DshAnimaMetrics {
    const float* losses;
    const float* step_ms;
    size_t step_count;
    size_t cache_hits;
    double prepare_seconds;
    double train_seconds;
    uint64_t peak_allocated_bytes;
    uint64_t peak_reserved_bytes;
    uint64_t prepare_peak_allocated_bytes;
    uint64_t prepare_peak_reserved_bytes;
    const char* device_name;
} DshAnimaMetrics;

// Callbacks and the trainer return zero on success. Input data from read_sample
// remains valid until the next read_sample call. Result/cache inputs are valid
// only during their callbacks. The error string is valid only during on_error.
typedef int (*DshAnimaReadSample)(void*, size_t, DshAnimaSample*);
typedef int (*DshAnimaWriteCache)(void*, size_t, const DshAnimaTensor*, const DshAnimaTensor*);
typedef int (*DshAnimaWriteOutput)(void*, const DshAnimaTensor*, size_t, const DshAnimaMetrics*);
typedef void (*DshAnimaError)(void*, const char*);

uint32_t dsh_anima_abi_version(void);
int dsh_anima_run(
    const DshAnimaConfig*, void*, DshAnimaReadSample, DshAnimaWriteCache, DshAnimaWriteOutput, DshAnimaError);

#ifdef __cplusplus
}
#endif

#endif
