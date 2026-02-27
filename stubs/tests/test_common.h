#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <string.h>

static int g_test_pass = 0;
static int g_test_fail = 0;

#define TEST_ASSERT(cond)                                               \
    do {                                                                \
        if (!(cond)) {                                                  \
            printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            g_test_fail++;                                              \
            return;                                                     \
        }                                                               \
    } while (0)

#define TEST_ASSERT_EQ_INT(a, b)                                            \
    do {                                                                    \
        int _a = (a), _b = (b);                                            \
        if (_a != _b) {                                                     \
            printf("  FAIL: %s == %s (%d != %d) (%s:%d)\n",                \
                   #a, #b, _a, _b, __FILE__, __LINE__);                     \
            g_test_fail++;                                                  \
            return;                                                         \
        }                                                                   \
    } while (0)

#define TEST_ASSERT_EQ_STR(a, b)                                            \
    do {                                                                    \
        const char *_a = (a), *_b = (b);                                    \
        if (strcmp(_a, _b) != 0) {                                          \
            printf("  FAIL: %s == %s (\"%s\" != \"%s\") (%s:%d)\n",        \
                   #a, #b, _a, _b, __FILE__, __LINE__);                     \
            g_test_fail++;                                                  \
            return;                                                         \
        }                                                                   \
    } while (0)

#define TEST_ASSERT_NOT_NULL(ptr)                                           \
    do {                                                                    \
        if ((ptr) == NULL) {                                                \
            printf("  FAIL: %s != NULL (%s:%d)\n", #ptr, __FILE__, __LINE__); \
            g_test_fail++;                                                  \
            return;                                                         \
        }                                                                   \
    } while (0)

#define TEST_ASSERT_NULL(ptr)                                               \
    do {                                                                    \
        if ((ptr) != NULL) {                                                \
            printf("  FAIL: %s == NULL (%s:%d)\n", #ptr, __FILE__, __LINE__); \
            g_test_fail++;                                                  \
            return;                                                         \
        }                                                                   \
    } while (0)

#define RUN_TEST(fn)                                \
    do {                                            \
        int _before = g_test_fail;                  \
        printf("  [RUN ] %s\n", #fn);              \
        fn();                                       \
        if (g_test_fail == _before) {               \
            printf("  [PASS] %s\n", #fn);           \
            g_test_pass++;                          \
        } else {                                    \
            printf("  [FAIL] %s\n", #fn);           \
        }                                           \
    } while (0)

#define TEST_SUMMARY()                                                      \
    do {                                                                    \
        printf("\n=== %d passed, %d failed ===\n", g_test_pass, g_test_fail); \
        return (g_test_fail > 0) ? 1 : 0;                                  \
    } while (0)

#endif /* TEST_COMMON_H */
