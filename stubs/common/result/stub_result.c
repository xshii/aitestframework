#include "stub_result.h"
#include "stub_memmap.h"
#include "stub_log.h"
#include <stdio.h>
#include <string.h>

#define CHUNK_SIZE 4096

int result_export(uint32_t addr, size_t size, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        LOG_ERROR("result_export: cannot open '%s' for writing", path);
        return -1;
    }

    uint8_t buf[CHUNK_SIZE];
    size_t remaining = size;
    uint32_t src = addr;

    while (remaining > 0) {
        size_t chunk = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
        int rc = memmap_read(src, buf, chunk);
        if (rc != 0) {
            LOG_ERROR("result_export: memmap_read failed at 0x%08X", src);
            fclose(fp);
            return rc;
        }
        if (fwrite(buf, 1, chunk, fp) != chunk) {
            LOG_ERROR("result_export: write error to '%s'", path);
            fclose(fp);
            return -1;
        }
        src += (uint32_t)chunk;
        remaining -= chunk;
    }

    fclose(fp);
    return 0;
}

int result_compare(uint32_t addr, size_t size, const char *golden_path,
                   result_mismatch_t *mm)
{
    FILE *fp = fopen(golden_path, "rb");
    if (!fp) {
        LOG_ERROR("result_compare: cannot open golden '%s'", golden_path);
        return -1;
    }

    /* verify golden file size */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize < 0 || (size_t)fsize != size) {
        LOG_ERROR("result_compare: golden size mismatch (%ld vs %zu)",
                  fsize, size);
        fclose(fp);
        return -1;
    }

    uint8_t mem_buf[CHUNK_SIZE];
    uint8_t gol_buf[CHUNK_SIZE];
    size_t remaining = size;
    uint32_t src = addr;
    size_t global_offset = 0;

    while (remaining > 0) {
        size_t chunk = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;

        int rc = memmap_read(src, mem_buf, chunk);
        if (rc != 0) {
            LOG_ERROR("result_compare: memmap_read failed at 0x%08X", src);
            fclose(fp);
            return -1;
        }

        if (fread(gol_buf, 1, chunk, fp) != chunk) {
            LOG_ERROR("result_compare: short read from golden '%s'",
                      golden_path);
            fclose(fp);
            return -1;
        }

        if (memcmp(mem_buf, gol_buf, chunk) != 0) {
            /* find first mismatch byte within this chunk */
            for (size_t j = 0; j < chunk; j++) {
                if (mem_buf[j] != gol_buf[j]) {
                    if (mm) {
                        mm->offset   = global_offset + j;
                        mm->actual   = mem_buf[j];
                        mm->expected = gol_buf[j];
                    }
                    fclose(fp);
                    return 1; /* mismatch */
                }
            }
        }

        src += (uint32_t)chunk;
        remaining -= chunk;
        global_offset += chunk;
    }

    fclose(fp);
    return 0; /* match */
}
