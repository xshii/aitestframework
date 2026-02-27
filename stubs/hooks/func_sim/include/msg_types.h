#ifndef MSG_TYPES_H
#define MSG_TYPES_H

#include <stdint.h>

typedef enum {
    MSG_WRITE_REG   = 0,
    MSG_READ_REG    = 1,
    MSG_START_COMP  = 2,
    MSG_STOP_COMP   = 3,
    MSG_DATA_XFER   = 4,
    MSG_STATUS_QUERY = 5
} msg_type_t;

typedef struct {
    msg_type_t type;
    uint32_t   addr;
    uint32_t   length;
    uint8_t    payload[256];
} platform_msg_t;

typedef struct {
    int      status;
    uint32_t length;
    uint8_t  data[256];
} platform_resp_t;

#endif /* MSG_TYPES_H */
