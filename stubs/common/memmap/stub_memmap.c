#include "stub_memmap.h"
#include "platform_api.h"
#include "stub_log.h"
#include <string.h>

static memmap_region_t s_regions[MEMMAP_MAX_REGIONS];
static int s_count = 0;

void memmap_reset(void)
{
    memset(s_regions, 0, sizeof(s_regions));
    s_count = 0;
}

/* Check whether [base1, base1+size1) overlaps with [base2, base2+size2) */
static int regions_overlap(uint32_t base1, uint32_t size1,
                           uint32_t base2, uint32_t size2)
{
    uint32_t end1 = base1 + size1;
    uint32_t end2 = base2 + size2;
    return (base1 < end2) && (base2 < end1);
}

int memmap_register(const char *name, uint32_t base, uint32_t size,
                    memmap_attr_t attr)
{
    if (s_count >= MEMMAP_MAX_REGIONS) {
        LOG_ERROR("memmap: region table full");
        return MEMMAP_ERR_FULL;
    }

    for (int i = 0; i < s_count; i++) {
        if (regions_overlap(s_regions[i].base, s_regions[i].size, base, size)) {
            LOG_ERROR("memmap: region '%s' [0x%08X,+0x%X) overlaps with '%s'",
                      name, base, size, s_regions[i].name);
            return MEMMAP_ERR_OVERLAP;
        }
    }

    memmap_region_t *r = &s_regions[s_count++];
    strncpy(r->name, name, sizeof(r->name) - 1);
    r->name[sizeof(r->name) - 1] = '\0';
    r->base = base;
    r->size = size;
    r->attr = attr;
    return MEMMAP_OK;
}

/* Find the region that fully contains [addr, addr+size) */
static const memmap_region_t *find_containing(uint32_t addr, size_t size)
{
    uint32_t end = addr + (uint32_t)size;
    for (int i = 0; i < s_count; i++) {
        uint32_t rbase = s_regions[i].base;
        uint32_t rend  = rbase + s_regions[i].size;
        if (addr >= rbase && end <= rend)
            return &s_regions[i];
    }
    return NULL;
}

int memmap_write(uint32_t addr, const void *data, size_t size)
{
    const memmap_region_t *r = find_containing(addr, size);
    if (!r) {
        LOG_ERROR("memmap_write: [0x%08X,+0x%zX) out of bounds", addr, size);
        return MEMMAP_ERR_OOB;
    }
    if (!(r->attr & MEM_W)) {
        LOG_ERROR("memmap_write: region '%s' not writable", r->name);
        return MEMMAP_ERR_PERM;
    }
    return platform_write_mem(addr, data, size);
}

int memmap_read(uint32_t addr, void *data, size_t size)
{
    const memmap_region_t *r = find_containing(addr, size);
    if (!r) {
        LOG_ERROR("memmap_read: [0x%08X,+0x%zX) out of bounds", addr, size);
        return MEMMAP_ERR_OOB;
    }
    if (!(r->attr & MEM_R)) {
        LOG_ERROR("memmap_read: region '%s' not readable", r->name);
        return MEMMAP_ERR_PERM;
    }
    return platform_read_mem(addr, data, size);
}

const memmap_region_t *memmap_find(const char *name)
{
    for (int i = 0; i < s_count; i++) {
        if (strcmp(s_regions[i].name, name) == 0)
            return &s_regions[i];
    }
    return NULL;
}

int memmap_count(void)
{
    return s_count;
}
