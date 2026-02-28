#include "stub_weight.h"
#include "stub_memmap.h"
#include "stub_log.h"
#include <stdio.h>
#include <string.h>

#define CHUNK_SIZE 4096

/* In-place byte-swap: reverse bytes within each word_size block */
static void swap_bytes(uint8_t *buf, size_t len, int word_size)
{
    for (size_t i = 0; i + (size_t)word_size <= len; i += (size_t)word_size) {
        for (int j = 0; j < word_size / 2; j++) {
            uint8_t tmp = buf[i + j];
            buf[i + j] = buf[i + word_size - 1 - j];
            buf[i + word_size - 1 - j] = tmp;
        }
    }
}

int weight_parse(const char *manifest_path, const char *base_dir,
                 weight_manifest_t *out)
{
    memset(out, 0, sizeof(*out));

    FILE *fp = fopen(manifest_path, "r");
    if (!fp) {
        LOG_ERROR("weight: cannot open manifest '%s'", manifest_path);
        return -1;
    }

    char line[512];
    weight_group_t *cur_group = NULL;

    while (fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';

        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '#') continue;

        /* count fields */
        char f1[256], f2[32];
        int nfields = sscanf(p, "%255s %31s", f1, f2);

        if (nfields == 2) {
            /* header line: base_addr  swap_word_size */
            if (out->group_count >= WEIGHT_MAX_GROUPS) {
                LOG_ERROR("weight: too many groups (max %d)", WEIGHT_MAX_GROUPS);
                fclose(fp);
                return -1;
            }
            unsigned int addr;
            int swap_w;
            if (sscanf(p, "%x %d", &addr, &swap_w) != 2) {
                LOG_ERROR("weight: bad header: %s", p);
                fclose(fp);
                return -1;
            }
            if (swap_w != 0 && swap_w != 2 && swap_w != 4 && swap_w != 8) {
                LOG_ERROR("weight: invalid swap %d (must be 0/2/4/8)", swap_w);
                fclose(fp);
                return -1;
            }
            cur_group = &out->groups[out->group_count++];
            cur_group->base_addr = (uint32_t)addr;
            cur_group->swap_word = swap_w;
            cur_group->start = out->item_count;
            cur_group->count = 0;
        } else if (nfields == 1) {
            /* entry line: bin_file */
            if (!cur_group) {
                LOG_ERROR("weight: entry before header: %s", p);
                fclose(fp);
                return -1;
            }
            if (out->item_count >= WEIGHT_MAX_TENSORS) {
                LOG_ERROR("weight: too many entries (max %d)", WEIGHT_MAX_TENSORS);
                fclose(fp);
                return -1;
            }
            weight_entry_t *e = &out->items[out->item_count++];
            if (base_dir && base_dir[0] != '\0' && f1[0] != '/') {
                snprintf(e->path, sizeof(e->path), "%s/%s", base_dir, f1);
            } else {
                strncpy(e->path, f1, sizeof(e->path) - 1);
                e->path[sizeof(e->path) - 1] = '\0';
            }
            cur_group->count++;
        } else {
            LOG_ERROR("weight: bad line: %s", p);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);

    if (out->group_count == 0) {
        LOG_ERROR("weight: manifest '%s' has no groups", manifest_path);
        return -1;
    }

    LOG_INFO("weight: parsed %d groups, %d entries from '%s'",
             out->group_count, out->item_count, manifest_path);
    return 0;
}

int weight_load_all(const weight_manifest_t *m)
{
    uint8_t buf[CHUNK_SIZE];

    for (int g = 0; g < m->group_count; g++) {
        const weight_group_t *grp = &m->groups[g];
        uint32_t dst = grp->base_addr;

        for (int i = grp->start; i < grp->start + grp->count; i++) {
            const weight_entry_t *e = &m->items[i];

            FILE *fp = fopen(e->path, "rb");
            if (!fp) {
                LOG_ERROR("weight: cannot open '%s'", e->path);
                return -1;
            }

            /* get file size */
            fseek(fp, 0, SEEK_END);
            long fsize = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (fsize <= 0) {
                LOG_ERROR("weight: empty or unreadable '%s'", e->path);
                fclose(fp);
                return -1;
            }
            size_t remaining = (size_t)fsize;

            while (remaining > 0) {
                size_t chunk = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
                size_t nread = fread(buf, 1, chunk, fp);
                if (nread != chunk) {
                    LOG_ERROR("weight: short read '%s'", e->path);
                    fclose(fp);
                    return -1;
                }
                if (grp->swap_word > 1)
                    swap_bytes(buf, nread, grp->swap_word);
                int rc = memmap_write(dst, buf, nread);
                if (rc != 0) {
                    LOG_ERROR("weight: memmap_write failed at 0x%08X", dst);
                    fclose(fp);
                    return rc;
                }
                dst += (uint32_t)nread;
                remaining -= nread;
            }

            fclose(fp);
            LOG_INFO("weight: loaded '%s' (%ld bytes -> 0x%08X%s)",
                     e->path, fsize, dst - (uint32_t)fsize,
                     grp->swap_word > 1 ? " [swapped]" : "");
        }
    }

    return 0;
}
