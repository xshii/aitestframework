#include "stub_config.h"
#include "stub_log.h"
#include "stub_memmap.h"
#include "stub_result.h"
#include "platform_api.h"
#include <string.h>
#include <stdio.h>

/*
 * FDD model layout:
 *   Group 1 weights: 0x2000, size 0x1000  (fdd_filter_w + fdd_filter_b = 40 bytes)
 *   Group 2 weights: 0x5000, size 0x1000  (fdd_dfe_w + fdd_dfe_b = 24 bytes, swap=4)
 *   output region:   0x8000, size 0x1000  (computation result, 64 bytes)
 */
#define FDD_WEIGHT1_BASE 0x2000
#define FDD_WEIGHT1_SIZE 0x1000
#define FDD_WEIGHT1_LEN  40   /* 32 + 8 */
#define FDD_WEIGHT2_BASE 0x5000
#define FDD_WEIGHT2_SIZE 0x1000
#define FDD_WEIGHT2_LEN  24   /* 16 + 8 */
#define FDD_OUTPUT_BASE  0x8000
#define FDD_OUTPUT_SIZE  0x1000
#define FDD_DATA_LEN     (FDD_WEIGHT1_LEN + FDD_WEIGHT2_LEN)  /* 64 */

int fdd_setup(const stub_config_t *cfg)
{
    (void)cfg;
    int rc;
    rc = memmap_register("fdd_filter", FDD_WEIGHT1_BASE, FDD_WEIGHT1_SIZE, MEM_RW);
    if (rc != 0) return rc;
    rc = memmap_register("fdd_dfe",    FDD_WEIGHT2_BASE, FDD_WEIGHT2_SIZE, MEM_RW);
    if (rc != 0) return rc;
    rc = memmap_register("fdd_output", FDD_OUTPUT_BASE,  FDD_OUTPUT_SIZE,  MEM_RW);
    return rc;
}

int fdd_run(const stub_config_t *cfg)
{
    /* 1. Read weights from both groups */
    uint8_t w1[FDD_WEIGHT1_LEN];
    if (memmap_read(FDD_WEIGHT1_BASE, w1, FDD_WEIGHT1_LEN) != 0) {
        LOG_ERROR("fdd: failed to read group 1 weights");
        return -1;
    }
    uint8_t w2[FDD_WEIGHT2_LEN];
    if (memmap_read(FDD_WEIGHT2_BASE, w2, FDD_WEIGHT2_LEN) != 0) {
        LOG_ERROR("fdd: failed to read group 2 weights");
        return -1;
    }

    /* 2. Compute: output[i] = weights[i] XOR 0xFF (trivial demo) */
    uint8_t output[FDD_DATA_LEN];
    for (int i = 0; i < FDD_WEIGHT1_LEN; i++)
        output[i] = w1[i] ^ 0xFF;
    for (int i = 0; i < FDD_WEIGHT2_LEN; i++)
        output[FDD_WEIGHT1_LEN + i] = w2[i] ^ 0xFF;

    /* 3. Write output */
    if (memmap_write(FDD_OUTPUT_BASE, output, FDD_DATA_LEN) != 0) {
        LOG_ERROR("fdd: failed to write output");
        return -1;
    }

    /* 4. Notify platform */
    platform_msg_t  msg  = {.type = MSG_DATA_XFER, .addr = FDD_OUTPUT_BASE, .length = FDD_DATA_LEN};
    platform_resp_t resp = {0};
    if (platform_send_msg(&msg, &resp) != 0) return -1;

    /* 5. Export output */
    if (cfg->result_dir[0] != '\0') {
        char path[512];
        snprintf(path, sizeof(path), "%s/fdd_output.bin", cfg->result_dir);
        if (result_export(FDD_OUTPUT_BASE, FDD_DATA_LEN, path) != 0) {
            LOG_ERROR("fdd: result export failed");
            return -1;
        }
        LOG_INFO("fdd: output exported to %s", path);
    }

    /* 6. Compare with golden */
    if (cfg->golden_dir[0] != '\0') {
        char golden_path[512];
        snprintf(golden_path, sizeof(golden_path), "%s/fdd_output.bin", cfg->golden_dir);
        result_mismatch_t mm;
        int cmp = result_compare(FDD_OUTPUT_BASE, FDD_DATA_LEN, golden_path, &mm);
        if (cmp == 1) {
            LOG_ERROR("fdd: MISMATCH at offset %zu (actual=0x%02X expected=0x%02X)",
                      mm.offset, mm.actual, mm.expected);
            return -1;
        } else if (cmp < 0) {
            LOG_ERROR("fdd: golden compare error (rc=%d)", cmp);
            return -1;
        }
        LOG_INFO("fdd: golden compare PASSED");
    }

    LOG_INFO("fdd: OK");
    return 0;
}
