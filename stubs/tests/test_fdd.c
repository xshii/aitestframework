#include "test_common.h"
#include "stub_registry.h"
#include "stub_config.h"
#include "stub_memmap.h"
#include "platform_api.h"

extern void platform_sim_setup(void);
extern const model_entry_t g_model_table[];
extern const int           g_model_table_count;

static void test_fdd_registered(void)
{
    TEST_ASSERT_EQ_INT(stub_registry_init(g_model_table, g_model_table_count), 0);
    const model_entry_t *m = stub_registry_find("fdd");
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQ_STR(m->name, "fdd");
    TEST_ASSERT_NOT_NULL(m->setup);
}

static void test_fdd_run(void)
{
    stub_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    const model_entry_t *m = stub_registry_find("fdd");
    TEST_ASSERT_NOT_NULL(m);

    memmap_reset();
    TEST_ASSERT_EQ_INT(m->setup(&cfg), 0);
    TEST_ASSERT_EQ_INT(m->run(&cfg), 0);
}

int main(void)
{
    platform_sim_setup();
    printf("=== test_fdd ===\n");
    RUN_TEST(test_fdd_registered);
    RUN_TEST(test_fdd_run);
    TEST_SUMMARY();
}
