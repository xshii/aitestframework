#include "test_common.h"
#include "stub_registry.h"
#include "stub_config.h"
#include "platform_api.h"

extern void platform_sim_setup(void);
extern const model_entry_t g_model_table[];
extern const int           g_model_table_count;

static void test_model_a_registered(void)
{
    TEST_ASSERT_EQ_INT(stub_registry_init(g_model_table, g_model_table_count), 0);
    const model_entry_t *m = stub_registry_find("demo_model_a");
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQ_STR(m->name, "demo_model_a");
}

static void test_model_a_run(void)
{
    stub_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    const model_entry_t *m = stub_registry_find("demo_model_a");
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQ_INT(m->run(&cfg), 0);
}

int main(void)
{
    platform_sim_setup();
    printf("=== test_demo_model_a ===\n");
    RUN_TEST(test_model_a_registered);
    RUN_TEST(test_model_a_run);
    TEST_SUMMARY();
}
