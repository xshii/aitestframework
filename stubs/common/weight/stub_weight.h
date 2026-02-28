#ifndef STUB_WEIGHT_H
#define STUB_WEIGHT_H

#include <stddef.h>
#include <stdint.h>

#define WEIGHT_MAX_TENSORS 128
#define WEIGHT_MAX_GROUPS  16

typedef struct {
    char path[256];
} weight_entry_t;

typedef struct {
    uint32_t base_addr;
    int      swap_word;   /* 0=none, 2/4/8=swap */
    int      start;       /* first entry index in items[] */
    int      count;       /* number of entries in this group */
} weight_group_t;

typedef struct {
    weight_entry_t items[WEIGHT_MAX_TENSORS];
    int            item_count;
    weight_group_t groups[WEIGHT_MAX_GROUPS];
    int            group_count;
} weight_manifest_t;

/*
 * Parse a manifest file.
 *
 * A manifest contains one or more groups. Each group starts with a header
 * line (2 fields) followed by entry lines (1 field):
 *
 *   base_addr  swap_word_size      ← header: start a new group
 *   bin_file                       ← entry (file size auto-detected)
 *   bin_file
 *   ...
 *   base_addr  swap_word_size      ← next group
 *   bin_file
 *   ...
 *
 * bin_file paths are resolved relative to base_dir.
 * Within each group, tensors are loaded sequentially from base_addr.
 *
 * Returns 0 on success, negative on error.
 */
int weight_parse(const char *manifest_path, const char *base_dir,
                 weight_manifest_t *out);

/*
 * Load all groups/tensors into memory via memmap_write.
 * File sizes are auto-detected at load time.
 *
 * Returns 0 on success, negative on error.
 */
int weight_load_all(const weight_manifest_t *m);

/*
 * (EMBED_WEIGHTS build option)
 * Load all tensors from data embedded at compile time.
 * No file I/O — data is baked into the binary by embed_weights.py.
 * Returns 0 on success, negative on error.
 */
#ifdef EMBED_WEIGHTS
int weight_load_embedded(void);
#endif

#endif /* STUB_WEIGHT_H */
