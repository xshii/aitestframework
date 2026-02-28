#include "stub_config.h"
#include "stub_log.h"
#include "stub_memmap.h"
#include "stub_result.h"
#include "platform_api.h"
#include <string.h>
#include <stdio.h>

/*
 * TDD model layout:
 *   weights region: 0x1000, size 0x1000  (input: tdd_conv_w + tdd_conv_b + tdd_fc_w = 56 bytes)
 *   output  region: 0x4000, size 0x1000  (output: computation result, 56 bytes)
 */
#define TDD_WEIGHT_BASE  0x1000
#define TDD_WEIGHT_SIZE  0x1000
#define TDD_OUTPUT_BASE  0x4000
#define TDD_OUTPUT_SIZE  0x1000
#define TDD_DATA_LEN     56  /* total weight bytes: 32 + 8 + 16 */

int tdd_setup(const stub_config_t *cfg)
{
    (void)cfg;
    int rc;
    rc = memmap_register("tdd_weights", TDD_WEIGHT_BASE, TDD_WEIGHT_SIZE, MEM_RW);
    if (rc != 0) return rc;
    rc = memmap_register("tdd_output",  TDD_OUTPUT_BASE, TDD_OUTPUT_SIZE, MEM_RW);
    return rc;
}

int tdd_run(const stub_config_t *cfg)
{
    /* 1. Read weights (already loaded by framework via weight_load_all) */
    uint8_t weights[TDD_DATA_LEN];
    if (memmap_read(TDD_WEIGHT_BASE, weights, TDD_DATA_LEN) != 0) {
        LOG_ERROR("tdd: failed to read weights");
        return -1;
    }

    /* 2. Compute: output[i] = weights[i] + 1 (trivial demo) */
    uint8_t output[TDD_DATA_LEN];
    for (int i = 0; i < TDD_DATA_LEN; i++)
        output[i] = (uint8_t)(weights[i] + 1);

    /* 3. Write output to output region */
    if (memmap_write(TDD_OUTPUT_BASE, output, TDD_DATA_LEN) != 0) {
        LOG_ERROR("tdd: failed to write output");
        return -1;
    }

    /* 4. Notify platform */
    platform_msg_t  msg  = {.type = MSG_START_COMP, .addr = TDD_OUTPUT_BASE, .length = TDD_DATA_LEN};
    platform_resp_t resp = {0};
    if (platform_send_msg(&msg, &resp) != 0) return -1;

    /* 5. Export output to file (if result_dir configured) */
    if (cfg->result_dir[0] != '\0') {
        char path[512];
        snprintf(path, sizeof(path), "%s/tdd_output.bin", cfg->result_dir);
        if (result_export(TDD_OUTPUT_BASE, TDD_DATA_LEN, path) != 0) {
            LOG_ERROR("tdd: result export failed");
            return -1;
        }
        LOG_INFO("tdd: output exported to %s", path);
    }

    /* 6. Compare with golden (if golden_dir configured) */
    if (cfg->golden_dir[0] != '\0') {
        char golden_path[512];
        snprintf(golden_path, sizeof(golden_path), "%s/tdd_output.bin", cfg->golden_dir);
        result_mismatch_t mm;
        int cmp = result_compare(TDD_OUTPUT_BASE, TDD_DATA_LEN, golden_path, &mm);
        if (cmp == 1) {
            LOG_ERROR("tdd: MISMATCH at offset %zu (actual=0x%02X expected=0x%02X)",
                      mm.offset, mm.actual, mm.expected);
            return -1;
        } else if (cmp < 0) {
            LOG_ERROR("tdd: golden compare error (rc=%d)", cmp);
            return -1;
        }
        LOG_INFO("tdd: golden compare PASSED");
    }

    LOG_INFO("tdd: OK");
    return 0;
}
