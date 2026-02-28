#include "test_common.h"
#include "stub_memmap.h"
#include "platform_api.h"

extern void platform_sim_setup(void);

static void test_register_and_find(void)
{
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_count(), 0);

    TEST_ASSERT_EQ_INT(memmap_register("region_a", 0x0000, 0x1000, MEM_RW), 0);
    TEST_ASSERT_EQ_INT(memmap_count(), 1);

    const memmap_region_t *r = memmap_find("region_a");
    TEST_ASSERT_NOT_NULL(r);
    TEST_ASSERT_EQ_STR(r->name, "region_a");
    TEST_ASSERT_EQ_INT((int)r->base, 0x0000);
    TEST_ASSERT_EQ_INT((int)r->size, 0x1000);
    TEST_ASSERT_EQ_INT((int)r->attr, MEM_RW);

    TEST_ASSERT_NULL(memmap_find("nonexistent"));
}

static void test_overlap_rejected(void)
{
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("r1", 0x1000, 0x1000, MEM_RW), 0);

    /* fully inside existing region */
    TEST_ASSERT_EQ_INT(memmap_register("r2", 0x1800, 0x100, MEM_RW),
                       MEMMAP_ERR_OVERLAP);

    /* partial overlap at start */
    TEST_ASSERT_EQ_INT(memmap_register("r3", 0x0800, 0x1000, MEM_RW),
                       MEMMAP_ERR_OVERLAP);

    /* no overlap — adjacent is OK */
    TEST_ASSERT_EQ_INT(memmap_register("r4", 0x2000, 0x1000, MEM_RW), 0);
    TEST_ASSERT_EQ_INT(memmap_count(), 2);
}

static void test_oob_rejected(void)
{
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("r1", 0x1000, 0x1000, MEM_RW), 0);

    uint8_t buf[4] = {0};
    /* completely outside */
    TEST_ASSERT_EQ_INT(memmap_write(0x0000, buf, 4), MEMMAP_ERR_OOB);
    TEST_ASSERT_EQ_INT(memmap_read(0x3000, buf, 4), MEMMAP_ERR_OOB);

    /* partially outside (crosses region end) */
    TEST_ASSERT_EQ_INT(memmap_write(0x1FFE, buf, 4), MEMMAP_ERR_OOB);
}

static void test_permission_check(void)
{
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("ro", 0x1000, 0x1000, MEM_R), 0);
    TEST_ASSERT_EQ_INT(memmap_register("wo", 0x2000, 0x1000, MEM_W), 0);

    uint8_t buf[4] = {1, 2, 3, 4};

    /* read-only: write should fail, read should succeed */
    TEST_ASSERT_EQ_INT(memmap_write(0x1000, buf, 4), MEMMAP_ERR_PERM);
    TEST_ASSERT_EQ_INT(memmap_read(0x1000, buf, 4), 0);

    /* write-only: read should fail, write should succeed */
    TEST_ASSERT_EQ_INT(memmap_write(0x2000, buf, 4), 0);
    TEST_ASSERT_EQ_INT(memmap_read(0x2000, buf, 4), MEMMAP_ERR_PERM);
}

static void test_read_write_roundtrip(void)
{
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("rw", 0x1000, 0x1000, MEM_RW), 0);

    uint8_t wr[8] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
    TEST_ASSERT_EQ_INT(memmap_write(0x1000, wr, sizeof(wr)), 0);

    uint8_t rd[8] = {0};
    TEST_ASSERT_EQ_INT(memmap_read(0x1000, rd, sizeof(rd)), 0);
    TEST_ASSERT(memcmp(wr, rd, sizeof(wr)) == 0);
}

static void test_reset(void)
{
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("a", 0x0000, 0x100, MEM_RW), 0);
    TEST_ASSERT_EQ_INT(memmap_register("b", 0x1000, 0x100, MEM_RW), 0);
    TEST_ASSERT_EQ_INT(memmap_count(), 2);

    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_count(), 0);
    TEST_ASSERT_NULL(memmap_find("a"));
}

int main(void)
{
    platform_sim_setup();
    printf("=== test_memmap ===\n");
    RUN_TEST(test_register_and_find);
    RUN_TEST(test_overlap_rejected);
    RUN_TEST(test_oob_rejected);
    RUN_TEST(test_permission_check);
    RUN_TEST(test_read_write_roundtrip);
    RUN_TEST(test_reset);
    TEST_SUMMARY();
}
