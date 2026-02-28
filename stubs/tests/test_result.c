#include "test_common.h"
#include "stub_result.h"
#include "stub_memmap.h"
#include "platform_api.h"

extern void platform_sim_setup(void);

static void write_file(const char *path, const void *data, size_t size)
{
    FILE *fp = fopen(path, "wb");
    if (fp) { fwrite(data, 1, size, fp); fclose(fp); }
}

static int read_file(const char *path, void *buf, size_t size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t n = fread(buf, 1, size, fp);
    fclose(fp);
    return (n == size) ? 0 : -1;
}

static void test_export(void)
{
    platform_sim_setup();
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("exp", 0x1000, 0x100, MEM_RW), 0);

    uint8_t data[16] = {0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44,
                        0x55, 0x66, 0x77, 0x88, 0x99, 0x00, 0xFF, 0xEE};
    TEST_ASSERT_EQ_INT(memmap_write(0x1000, data, sizeof(data)), 0);

    TEST_ASSERT_EQ_INT(result_export(0x1000, sizeof(data), "/tmp/test_export.bin"), 0);

    uint8_t rb[16];
    TEST_ASSERT_EQ_INT(read_file("/tmp/test_export.bin", rb, sizeof(rb)), 0);
    TEST_ASSERT(memcmp(data, rb, sizeof(data)) == 0);
}

static void test_compare_match(void)
{
    platform_sim_setup();
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("cmp", 0x1000, 0x100, MEM_RW), 0);

    uint8_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    TEST_ASSERT_EQ_INT(memmap_write(0x1000, data, sizeof(data)), 0);

    /* write identical golden file */
    write_file("/tmp/test_golden_ok.bin", data, sizeof(data));

    result_mismatch_t mm = {0};
    TEST_ASSERT_EQ_INT(result_compare(0x1000, sizeof(data),
                                      "/tmp/test_golden_ok.bin", &mm), 0);
}

static void test_compare_mismatch(void)
{
    platform_sim_setup();
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("cmp2", 0x1000, 0x100, MEM_RW), 0);

    uint8_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    TEST_ASSERT_EQ_INT(memmap_write(0x1000, data, sizeof(data)), 0);

    /* golden with difference at byte 3 */
    uint8_t golden[8] = {1, 2, 3, 99, 5, 6, 7, 8};
    write_file("/tmp/test_golden_bad.bin", golden, sizeof(golden));

    result_mismatch_t mm = {0};
    int rc = result_compare(0x1000, sizeof(data), "/tmp/test_golden_bad.bin", &mm);
    TEST_ASSERT_EQ_INT(rc, 1); /* mismatch */
    TEST_ASSERT_EQ_INT((int)mm.offset, 3);
    TEST_ASSERT_EQ_INT(mm.actual, 4);
    TEST_ASSERT_EQ_INT(mm.expected, 99);
}

int main(void)
{
    platform_sim_setup();
    printf("=== test_result ===\n");
    RUN_TEST(test_export);
    RUN_TEST(test_compare_match);
    RUN_TEST(test_compare_mismatch);
    TEST_SUMMARY();
}
