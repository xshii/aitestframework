# ============================================================
# Model discovery + registry table generation
# ============================================================

# Compile-time default run list: -DSTUB_MODELS="tdd;fdd"
# Empty = run all models by default
set(STUB_MODELS "" CACHE STRING "Default models to run (empty = all)")

set(ALL_MODEL_NAMES     "")
set(ALL_MODEL_RUN_FNS   "")
set(ALL_MODEL_SETUP_FNS "")
set(ALL_MODEL_SOURCES   "")

# Auto-discover model subdirectories (any dir with CMakeLists.txt)
file(GLOB _model_cmakes CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt
)

foreach(_cmake_file ${_model_cmakes})
    get_filename_component(_dir ${_cmake_file} DIRECTORY)
    get_filename_component(_name ${_dir} NAME)

    set(MODEL_SETUP_FN "" CACHE INTERNAL "" FORCE)
    add_subdirectory(${_name})
    list(APPEND ALL_MODEL_NAMES     "${MODEL_NAME}")
    list(APPEND ALL_MODEL_RUN_FNS   "${MODEL_RUN_FN}")
    list(APPEND ALL_MODEL_SETUP_FNS "${MODEL_SETUP_FN}")

    file(GLOB_RECURSE _srcs CONFIGURE_DEPENDS ${_dir}/*.c)
    list(APPEND ALL_MODEL_SOURCES ${_srcs})
    message(STATUS "Include model: ${_name}")
endforeach()

# --- Generate stub_model_table.c ---
set(MODEL_EXTERN_DECLS "")
set(MODEL_TABLE_ENTRIES "")
list(LENGTH ALL_MODEL_NAMES MODEL_COUNT)

set(MODEL_SETUP_EXTERN_DECLS "")

if(MODEL_COUNT GREATER 0)
    math(EXPR MODEL_LAST "${MODEL_COUNT} - 1")
    foreach(idx RANGE ${MODEL_LAST})
        list(GET ALL_MODEL_NAMES     ${idx} _n)
        list(GET ALL_MODEL_RUN_FNS   ${idx} _fn)
        list(GET ALL_MODEL_SETUP_FNS ${idx} _sfn)
        string(APPEND MODEL_EXTERN_DECLS "extern int ${_fn}(const stub_config_t *cfg);\n")
        if(_sfn)
            string(APPEND MODEL_SETUP_EXTERN_DECLS "extern int ${_sfn}(const stub_config_t *cfg);\n")
            string(APPEND MODEL_TABLE_ENTRIES "    {\"${_n}\", ${_fn}, ${_sfn}},\n")
        else()
            string(APPEND MODEL_TABLE_ENTRIES "    {\"${_n}\", ${_fn}, NULL},\n")
        endif()
    endforeach()
endif()

# --- Default run list (compile-time selection) ---
set(RUN_LIST_ENTRIES "")
set(RUN_LIST_COUNT 0)
if(STUB_MODELS)
    foreach(_run_name ${STUB_MODELS})
        string(APPEND RUN_LIST_ENTRIES "    \"${_run_name}\",\n")
    endforeach()
    list(LENGTH STUB_MODELS RUN_LIST_COUNT)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/stub_model_table.c.in
    ${CMAKE_CURRENT_BINARY_DIR}/stub_model_table.c
    @ONLY
)

set(ALL_MODEL_SOURCES "${ALL_MODEL_SOURCES}" PARENT_SCOPE)
set(GENERATED_MODEL_TABLE "${CMAKE_CURRENT_BINARY_DIR}/stub_model_table.c" PARENT_SCOPE)
set(MODEL_COUNT "${MODEL_COUNT}" PARENT_SCOPE)
