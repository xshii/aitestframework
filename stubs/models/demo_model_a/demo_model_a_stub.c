#include "stub_config.h"
#include "stub_log.h"
#include "platform_api.h"
#include <string.h>

int demo_model_a_run(const stub_config_t *cfg)
{
    (void)cfg;

    uint8_t data[16] = {1, 2, 3, 4, 5, 6, 7, 8};

    if (platform_write_mem(0x1000, data, sizeof(data)) != 0) return -1;

    platform_msg_t  msg  = {.type = MSG_START_COMP, .addr = 0x1000, .length = sizeof(data)};
    platform_resp_t resp = {0};
    if (platform_send_msg(&msg, &resp) != 0) return -1;

    uint8_t rb[16];
    if (platform_read_mem(0x1000, rb, sizeof(rb)) != 0) return -1;
    if (memcmp(rb, data, sizeof(data)) != 0)             return -1;

    /* TODO: add real model-specific logic here */

    LOG_INFO("demo_model_a: OK");
    return 0;
}
