#include "stub_registry.h"
#include "stub_config.h"
#include "stub_log.h"
#include "platform_api.h"
#include <string.h>

extern const model_entry_t g_model_table[];
extern const int           g_model_table_count;

/* Hardcoded entry point — platform calls this after registering hooks */
int stub_entry(int argc, char *argv[])
{
    stub_config_t cfg;
    int rc = stub_config_parse(argc, argv, &cfg);
    if (rc != 0) return (rc > 0) ? 0 : 1;

    stub_registry_init(g_model_table, g_model_table_count);

    if (cfg.list_models) {
        const model_entry_t *all = stub_registry_get_all();
        for (int i = 0; i < stub_registry_count(); i++)
            printf("  %s\n", all[i].name);
        return 0;
    }

    int n = stub_registry_count();
    const model_entry_t *all = stub_registry_get_all();
    int failed = 0;

    for (int i = 0; i < n; i++) {
        /* If --models given, skip unselected */
        if (cfg.model_count > 0) {
            int found = 0;
            for (int j = 0; j < cfg.model_count; j++)
                if (strcmp(cfg.model_names[j], all[i].name) == 0) { found = 1; break; }
            if (!found) continue;
        }
        LOG_INFO("running: %s", all[i].name);
        rc = all[i].run(&cfg);
        if (rc != 0) { LOG_ERROR("%s FAILED (rc=%d)", all[i].name, rc); failed++; }
        else         { LOG_INFO("%s PASSED", all[i].name); }
    }
    return failed ? 1 : 0;
}
