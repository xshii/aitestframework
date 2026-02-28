#ifndef STUB_MEMMAP_H
#define STUB_MEMMAP_H

#include <stddef.h>
#include <stdint.h>

#define MEMMAP_MAX_REGIONS 32

/* Error codes */
#define MEMMAP_OK        0
#define MEMMAP_ERR_FULL  (-1)
#define MEMMAP_ERR_OVERLAP (-2)
#define MEMMAP_ERR_OOB   (-3)  /* out of bounds */
#define MEMMAP_ERR_PERM  (-4)  /* permission denied */

typedef enum { MEM_R = 1, MEM_W = 2, MEM_RW = 3 } memmap_attr_t;

typedef struct {
    char          name[32];
    uint32_t      base;
    uint32_t      size;
    memmap_attr_t attr;
} memmap_region_t;

/* Clear all registered regions */
void memmap_reset(void);

/* Register a new memory region; returns 0 on success, negative on error */
int memmap_register(const char *name, uint32_t base, uint32_t size,
                    memmap_attr_t attr);

/* Write data through platform_write_mem with bounds + permission check */
int memmap_write(uint32_t addr, const void *data, size_t size);

/* Read data through platform_read_mem with bounds + permission check */
int memmap_read(uint32_t addr, void *data, size_t size);

/* Find a region by name; returns NULL if not found */
const memmap_region_t *memmap_find(const char *name);

/* Return number of registered regions */
int memmap_count(void);

#endif /* STUB_MEMMAP_H */
