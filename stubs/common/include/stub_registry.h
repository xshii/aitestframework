#ifndef STUB_REGISTRY_H
#define STUB_REGISTRY_H

#include "stub_config.h"

/* Model run function signature */
typedef int (*model_run_fn)(const stub_config_t *cfg);

/* Model entry in the registry */
typedef struct {
    const char  *name;
    model_run_fn run;
} model_entry_t;

/* Initialise registry from an externally-provided table (CMake-generated) */
int stub_registry_init(const model_entry_t *table, int count);

/* Find a model by name; returns NULL if not found */
const model_entry_t *stub_registry_find(const char *name);

/* Return the number of registered models */
int stub_registry_count(void);

/* Return pointer to the internal array of entries */
const model_entry_t *stub_registry_get_all(void);

#endif /* STUB_REGISTRY_H */
