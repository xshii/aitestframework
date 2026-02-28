#include "test_common.h"
#include "stub_weight.h"
#include "stub_memmap.h"
#include "platform_api.h"
#include <stdlib.h>

extern void platform_sim_setup(void);

static void write_file(const char *path, const void *data, size_t size)
{
    FILE *fp = fopen(path, "wb");
    if (fp) { fwrite(data, 1, size, fp); fclose(fp); }
}

static void test_parse_single_group(void)
{
    const char *manifest =
        "# base  swap\n"
        "0x1000  0\n"
        "a.bin\n"
        "b.bin\n";

    write_file("/tmp/test_wm.txt", manifest, strlen(manifest));

    weight_manifest_t wm;
    TEST_ASSERT_EQ_INT(weight_parse("/tmp/test_wm.txt", "/tmp", &wm), 0);
    TEST_ASSERT_EQ_INT(wm.group_count, 1);
    TEST_ASSERT_EQ_INT(wm.item_count, 2);
    TEST_ASSERT_EQ_INT((int)wm.groups[0].base_addr, 0x1000);
    TEST_ASSERT_EQ_INT(wm.groups[0].swap_word, 0);
    TEST_ASSERT_EQ_INT(wm.groups[0].start, 0);
    TEST_ASSERT_EQ_INT(wm.groups[0].count, 2);
    TEST_ASSERT_EQ_STR(wm.items[0].path, "/tmp/a.bin");
    TEST_ASSERT_EQ_STR(wm.items[1].path, "/tmp/b.bin");
}

static void test_parse_multi_group(void)
{
    const char *manifest =
        "0x1000  0\n"
        "g1a.bin\n"
        "g1b.bin\n"
        "\n"
        "0x5000  4\n"
        "g2a.bin\n";

    write_file("/tmp/test_wm2.txt", manifest, strlen(manifest));

    weight_manifest_t wm;
    TEST_ASSERT_EQ_INT(weight_parse("/tmp/test_wm2.txt", "/tmp", &wm), 0);
    TEST_ASSERT_EQ_INT(wm.group_count, 2);
    TEST_ASSERT_EQ_INT(wm.item_count, 3);

    TEST_ASSERT_EQ_INT((int)wm.groups[0].base_addr, 0x1000);
    TEST_ASSERT_EQ_INT(wm.groups[0].count, 2);
    TEST_ASSERT_EQ_INT((int)wm.groups[1].base_addr, 0x5000);
    TEST_ASSERT_EQ_INT(wm.groups[1].swap_word, 4);
    TEST_ASSERT_EQ_INT(wm.groups[1].count, 1);
}

static void test_load_sequential(void)
{
    platform_sim_setup();
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("r1", 0x1000, 0x100, MEM_RW), 0);

    uint8_t da[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t db[4] = {0xA0, 0xB0, 0xC0, 0xD0};
    write_file("/tmp/tw_a.bin", da, sizeof(da));
    write_file("/tmp/tw_b.bin", db, sizeof(db));

    const char *manifest = "0x1000  0\ntw_a.bin\ntw_b.bin\n";
    write_file("/tmp/tw_mf.txt", manifest, strlen(manifest));

    weight_manifest_t wm;
    TEST_ASSERT_EQ_INT(weight_parse("/tmp/tw_mf.txt", "/tmp", &wm), 0);
    TEST_ASSERT_EQ_INT(weight_load_all(&wm), 0);

    uint8_t rb[8];
    TEST_ASSERT_EQ_INT(memmap_read(0x1000, rb, 8), 0);
    TEST_ASSERT(memcmp(rb, da, 8) == 0);

    /* second file at 0x1000 + 8 = 0x1008 */
    uint8_t rb2[4];
    TEST_ASSERT_EQ_INT(memmap_read(0x1008, rb2, 4), 0);
    TEST_ASSERT(memcmp(rb2, db, 4) == 0);
}

static void test_load_multi_group(void)
{
    platform_sim_setup();
    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("r1", 0x1000, 0x100, MEM_RW), 0);
    TEST_ASSERT_EQ_INT(memmap_register("r2", 0x5000, 0x100, MEM_RW), 0);

    uint8_t d1[4] = {0x11, 0x22, 0x33, 0x44};
    uint8_t d2[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    write_file("/tmp/tw_g1.bin", d1, sizeof(d1));
    write_file("/tmp/tw_g2.bin", d2, sizeof(d2));

    const char *manifest =
        "0x1000  0\n"
        "tw_g1.bin\n"
        "0x5000  4\n"
        "tw_g2.bin\n";
    write_file("/tmp/tw_mg.txt", manifest, strlen(manifest));

    weight_manifest_t wm;
    TEST_ASSERT_EQ_INT(weight_parse("/tmp/tw_mg.txt", "/tmp", &wm), 0);
    TEST_ASSERT_EQ_INT(weight_load_all(&wm), 0);

    uint8_t rb1[4];
    TEST_ASSERT_EQ_INT(memmap_read(0x1000, rb1, 4), 0);
    TEST_ASSERT(memcmp(rb1, d1, 4) == 0);

    /* group 2 has swap=4, so 0xDEADBEEF -> 0xEFBEADDE */
    uint8_t expected[4] = {0xEF, 0xBE, 0xAD, 0xDE};
    uint8_t rb2[4];
    TEST_ASSERT_EQ_INT(memmap_read(0x5000, rb2, 4), 0);
    TEST_ASSERT(memcmp(rb2, expected, 4) == 0);
}

static void test_missing_file(void)
{
    const char *manifest = "0x1000  0\nnonexistent.bin\n";
    write_file("/tmp/tw_miss.txt", manifest, strlen(manifest));

    weight_manifest_t wm;
    TEST_ASSERT_EQ_INT(weight_parse("/tmp/tw_miss.txt", "/tmp", &wm), 0);

    memmap_reset();
    TEST_ASSERT_EQ_INT(memmap_register("r1", 0x1000, 0x100, MEM_RW), 0);
    TEST_ASSERT(weight_load_all(&wm) != 0);
}

int main(void)
{
    platform_sim_setup();
    printf("=== test_weight ===\n");
    RUN_TEST(test_parse_single_group);
    RUN_TEST(test_parse_multi_group);
    RUN_TEST(test_load_sequential);
    RUN_TEST(test_load_multi_group);
    RUN_TEST(test_missing_file);
    TEST_SUMMARY();
}
