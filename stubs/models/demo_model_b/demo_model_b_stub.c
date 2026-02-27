#include "stub_config.h"
#include "stub_log.h"
#include "platform_api.h"
#include <string.h>

int demo_model_b_run(const stub_config_t *cfg)
{
    (void)cfg;

    uint8_t weights[32];
    for (int i = 0; i < 32; i++) weights[i] = (uint8_t)(0xA0 + i);

    if (platform_write_mem(0x2000, weights, sizeof(weights)) != 0) return -1;

    platform_msg_t  msg  = {.type = MSG_DATA_XFER, .addr = 0x2000, .length = sizeof(weights)};
    platform_resp_t resp = {0};
    if (platform_send_msg(&msg, &resp) != 0) return -1;

    uint8_t rb[32];
    if (platform_read_mem(0x2000, rb, sizeof(rb)) != 0)  return -1;
    if (memcmp(rb, weights, sizeof(weights)) != 0)        return -1;

    /* TODO: add real model-specific logic here */

    LOG_INFO("demo_model_b: OK");
    return 0;
}
