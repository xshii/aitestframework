#include "stub_config.h"
#include <stdio.h>
#include <string.h>

int stub_config_parse(int argc, char *argv[], stub_config_t *cfg)
{
    memset(cfg, 0, sizeof(*cfg));

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            char buf[1024];
            strncpy(buf, argv[++i], sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            for (char *tok = strtok(buf, ","); tok && cfg->model_count < MAX_MODEL_NAMES; tok = strtok(NULL, ",")) {
                strncpy(cfg->model_names[cfg->model_count], tok, 63);
                cfg->model_names[cfg->model_count++][63] = '\0';
            }
        } else if (strcmp(argv[i], "--weight-manifest") == 0 && i + 1 < argc) {
            strncpy(cfg->weight_manifest, argv[++i], sizeof(cfg->weight_manifest) - 1);
            cfg->weight_manifest[sizeof(cfg->weight_manifest) - 1] = '\0';
        } else if (strcmp(argv[i], "--weight-dir") == 0 && i + 1 < argc) {
            strncpy(cfg->weight_dir, argv[++i], sizeof(cfg->weight_dir) - 1);
            cfg->weight_dir[sizeof(cfg->weight_dir) - 1] = '\0';
        } else if (strcmp(argv[i], "--result-dir") == 0 && i + 1 < argc) {
            strncpy(cfg->result_dir, argv[++i], sizeof(cfg->result_dir) - 1);
            cfg->result_dir[sizeof(cfg->result_dir) - 1] = '\0';
        } else if (strcmp(argv[i], "--golden-dir") == 0 && i + 1 < argc) {
            strncpy(cfg->golden_dir, argv[++i], sizeof(cfg->golden_dir) - 1);
            cfg->golden_dir[sizeof(cfg->golden_dir) - 1] = '\0';
        } else if (strcmp(argv[i], "--list") == 0) {
            cfg->list_models = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--models a,b,...] [--list]\n"
                   "       [--weight-manifest <path>] [--weight-dir <dir>]\n"
                   "       [--result-dir <dir>] [--golden-dir <dir>]\n"
                   "       [--help]\n", argv[0]);
            return 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}
