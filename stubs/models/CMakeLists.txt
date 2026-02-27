# ============================================================
# Auto-discover model subdirectories and generate registry table
# ============================================================

set(ALL_MODEL_NAMES   "")
set(ALL_MODEL_RUN_FNS "")
set(ALL_MODEL_SOURCES "")

# --- Add each model subdirectory here ---
add_subdirectory(demo_model_a)
list(APPEND ALL_MODEL_NAMES   "${MODEL_NAME}")
list(APPEND ALL_MODEL_RUN_FNS "${MODEL_RUN_FN}")
list(APPEND ALL_MODEL_SOURCES "${MODEL_SOURCE}")

add_subdirectory(demo_model_b)
list(APPEND ALL_MODEL_NAMES   "${MODEL_NAME}")
list(APPEND ALL_MODEL_RUN_FNS "${MODEL_RUN_FN}")
list(APPEND ALL_MODEL_SOURCES "${MODEL_SOURCE}")

# --- Generate stub_model_table.c via configure_file ---

# Build the extern declarations and table entries
set(MODEL_EXTERN_DECLS "")
set(MODEL_TABLE_ENTRIES "")
list(LENGTH ALL_MODEL_NAMES MODEL_COUNT)

math(EXPR MODEL_LAST "${MODEL_COUNT} - 1")
foreach(idx RANGE ${MODEL_LAST})
    list(GET ALL_MODEL_NAMES   ${idx} _name)
    list(GET ALL_MODEL_RUN_FNS ${idx} _fn)
    string(APPEND MODEL_EXTERN_DECLS "extern int ${_fn}(const stub_config_t *cfg);\n")
    string(APPEND MODEL_TABLE_ENTRIES "    {\"${_name}\", ${_fn}},\n")
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/stub_model_table.c.in
    ${CMAKE_CURRENT_BINARY_DIR}/stub_model_table.c
    @ONLY
)

# Export variables to parent scope for use by top-level CMakeLists.txt
set(ALL_MODEL_SOURCES "${ALL_MODEL_SOURCES}" PARENT_SCOPE)
set(GENERATED_MODEL_TABLE "${CMAKE_CURRENT_BINARY_DIR}/stub_model_table.c" PARENT_SCOPE)
set(MODEL_COUNT "${MODEL_COUNT}" PARENT_SCOPE)
