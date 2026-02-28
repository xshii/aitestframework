#include "platform_api.h"

static pfn_send_msg  g_send_msg;
static pfn_write_mem g_write_mem;
static pfn_read_mem  g_read_mem;
static pfn_stop_case g_stop_case;
static int           g_registered = 0;

void platform_register(pfn_send_msg  send_msg,
                       pfn_write_mem write_mem,
                       pfn_read_mem  read_mem,
                       pfn_stop_case stop_case)
{
    g_send_msg  = send_msg;
    g_write_mem = write_mem;
    g_read_mem  = read_mem;
    g_stop_case = stop_case;
    g_registered = 1;
}

int platform_send_msg(const platform_msg_t *msg, platform_resp_t *resp)
{
    if (!g_registered || !g_send_msg) return -1;
    return g_send_msg(msg, resp);
}

int platform_write_mem(uint32_t addr, const void *data, size_t size)
{
    if (!g_registered || !g_write_mem) return -1;
    return g_write_mem(addr, data, size);
}

int platform_read_mem(uint32_t addr, void *data, size_t size)
{
    if (!g_registered || !g_read_mem) return -1;
    return g_read_mem(addr, data, size);
}

int platform_stop_case(int reason)
{
    if (!g_registered || !g_stop_case) return -1;
    return g_stop_case(reason);
}
