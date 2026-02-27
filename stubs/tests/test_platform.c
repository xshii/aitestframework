#include "test_common.h"
#include "platform_api.h"

extern void platform_sim_setup(void);

static void test_write_read_mem(void)
{
    uint8_t wr[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t rd[8] = {0};
    TEST_ASSERT_EQ_INT(platform_write_mem(0x100, wr, sizeof(wr)), 0);
    TEST_ASSERT_EQ_INT(platform_read_mem(0x100, rd, sizeof(rd)), 0);
    TEST_ASSERT(memcmp(wr, rd, sizeof(wr)) == 0);
}

static void test_send_msg(void)
{
    platform_msg_t  msg  = {.type = MSG_START_COMP};
    platform_resp_t resp = {0};
    TEST_ASSERT_EQ_INT(platform_send_msg(&msg, &resp), 0);
    TEST_ASSERT_EQ_INT(resp.status, 0);
}

static void test_stop_case(void)
{
    TEST_ASSERT_EQ_INT(platform_stop_case(0), 0);
}

int main(void)
{
    platform_sim_setup();
    printf("=== test_platform ===\n");
    RUN_TEST(test_write_read_mem);
    RUN_TEST(test_send_msg);
    RUN_TEST(test_stop_case);
    TEST_SUMMARY();
}
