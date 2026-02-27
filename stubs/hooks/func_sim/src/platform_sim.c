#include "platform_api.h"
#include <stdio.h>
#include <string.h>

#define DDR_SIZE (64 * 1024) /* 64 KB simulated DDR */

static uint8_t s_ddr[DDR_SIZE];

static int sim_send_msg(const platform_msg_t *msg, platform_resp_t *resp)
{
    if (!msg || !resp) return -1;
    printf("[sim] send_msg: type=%d addr=0x%08X len=%u\n",
           msg->type, msg->addr, msg->length);
    resp->status = 0;
    resp->length = 0;
    memset(resp->data, 0, sizeof(resp->data));
    return 0;
}

static int sim_write_mem(uint32_t addr, const void *data, size_t size)
{
    if (!data || size == 0) return -1;
    if ((size_t)addr + size > DDR_SIZE) return -1;
    memcpy(&s_ddr[addr], data, size);
    return 0;
}

static int sim_read_mem(uint32_t addr, void *data, size_t size)
{
    if (!data || size == 0) return -1;
    if ((size_t)addr + size > DDR_SIZE) return -1;
    memcpy(data, &s_ddr[addr], size);
    return 0;
}

static int sim_stop_case(int reason)
{
    printf("[sim] stop_case: reason=%d\n", reason);
    return 0;
}

void platform_sim_setup(void)
{
    memset(s_ddr, 0, DDR_SIZE);
    platform_hooks_t hooks = {
        .send_msg  = sim_send_msg,
        .write_mem = sim_write_mem,
        .read_mem  = sim_read_mem,
        .stop_case = sim_stop_case,
    };
    platform_register(&hooks);
}
