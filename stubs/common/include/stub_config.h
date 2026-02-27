#ifndef STUB_CONFIG_H
#define STUB_CONFIG_H

#define MAX_MODEL_NAMES 64

typedef struct stub_config {
    char model_names[MAX_MODEL_NAMES][64];
    int  model_count;
    int  list_models;
} stub_config_t;

int stub_config_parse(int argc, char *argv[], stub_config_t *cfg);

#endif /* STUB_CONFIG_H */
