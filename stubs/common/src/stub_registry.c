#include "stub_registry.h"
#include <string.h>

static const model_entry_t *s_table = NULL;
static int                  s_count = 0;

int stub_registry_init(const model_entry_t *table, int count)
{
    s_table = table;
    s_count = count;
    return (table && count > 0) ? 0 : -1;
}

const model_entry_t *stub_registry_find(const char *name)
{
    for (int i = 0; i < s_count; i++)
        if (strcmp(s_table[i].name, name) == 0) return &s_table[i];
    return NULL;
}

int stub_registry_count(void) { return s_count; }
const model_entry_t *stub_registry_get_all(void) { return s_table; }
