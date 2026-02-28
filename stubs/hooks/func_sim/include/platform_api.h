#ifndef PLATFORM_API_H
#define PLATFORM_API_H

#include "msg_types.h"
#include <stddef.h>
#include <stdint.h>

/* ---- Hook function pointer types ---- */
typedef int (*pfn_send_msg)(const platform_msg_t *msg, platform_resp_t *resp);
typedef int (*pfn_write_mem)(uint32_t addr, const void *data, size_t size);
typedef int (*pfn_read_mem)(uint32_t addr, void *data, size_t size);
typedef int (*pfn_stop_case)(int reason);

/* Register hooks — platform calls this with all hooks as flat args */
void platform_register(pfn_send_msg  send_msg,
                       pfn_write_mem write_mem,
                       pfn_read_mem  read_mem,
                       pfn_stop_case stop_case);

/* Convenience wrappers — call through registered hooks */
int platform_send_msg(const platform_msg_t *msg, platform_resp_t *resp);
int platform_write_mem(uint32_t addr, const void *data, size_t size);
int platform_read_mem(uint32_t addr, void *data, size_t size);
int platform_stop_case(int reason);

#endif /* PLATFORM_API_H */
