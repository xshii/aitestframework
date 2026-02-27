#ifndef STUB_LOG_H
#define STUB_LOG_H

#include <stdio.h>

#define LOG_INFO(fmt, ...)  printf("[INFO]  " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  printf("[WARN]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#endif /* STUB_LOG_H */
