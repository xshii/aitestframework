#ifndef STUB_RESULT_H
#define STUB_RESULT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t  offset;     /* byte offset of first mismatch within the region */
    uint8_t actual;
    uint8_t expected;
} result_mismatch_t;

/*
 * Export memory region [addr, addr+size) to a binary file.
 * Returns 0 on success, negative on error.
 */
int result_export(uint32_t addr, size_t size, const char *path);

/*
 * Compare memory region [addr, addr+size) with golden binary file.
 * On mismatch, fills *mm with the first differing byte info.
 * Returns 0 = match, 1 = mismatch, negative = error.
 */
int result_compare(uint32_t addr, size_t size, const char *golden_path,
                   result_mismatch_t *mm);

#endif /* STUB_RESULT_H */
