#include "platform_api.h"
#include <string.h>

static platform_hooks_t g_hooks;
static int              g_registered = 0;

void platform_register(const platform_hooks_t *hooks)
{
    if (hooks) {
        memcpy(&g_hooks, hooks, sizeof(g_hooks));
        g_registered = 1;
    }
}

int platform_send_msg(const platform_msg_t *msg, platform_resp_t *resp)
{
    if (!g_registered || !g_hooks.send_msg) return -1;
    return g_hooks.send_msg(msg, resp);
}

int platform_write_mem(uint32_t addr, const void *data, size_t size)
{
    if (!g_registered || !g_hooks.write_mem) return -1;
    return g_hooks.write_mem(addr, data, size);
}

int platform_read_mem(uint32_t addr, void *data, size_t size)
{
    if (!g_registered || !g_hooks.read_mem) return -1;
    return g_hooks.read_mem(addr, data, size);
}

int platform_stop_case(int reason)
{
    if (!g_registered || !g_hooks.stop_case) return -1;
    return g_hooks.stop_case(reason);
}
