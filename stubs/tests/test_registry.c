#include "test_common.h"
#include "stub_registry.h"

static int dummy_run(const stub_config_t *cfg) { (void)cfg; return 0; }

static const model_entry_t test_table[] = {
    {"test_a", dummy_run, NULL},
    {"test_b", dummy_run, NULL},
};

static void test_init_and_count(void)
{
    TEST_ASSERT_EQ_INT(stub_registry_init(test_table, 2), 0);
    TEST_ASSERT_EQ_INT(stub_registry_count(), 2);
}

static void test_find(void)
{
    const model_entry_t *m = stub_registry_find("test_a");
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQ_STR(m->name, "test_a");

    TEST_ASSERT_NULL(stub_registry_find("nonexistent"));
}

int main(void)
{
    printf("=== test_registry ===\n");
    RUN_TEST(test_init_and_count);
    RUN_TEST(test_find);
    TEST_SUMMARY();
}
