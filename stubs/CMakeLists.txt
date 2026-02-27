cmake_minimum_required(VERSION 3.10)
project(stub_framework C)

# ============================================================
# Options
# ============================================================
set(HOOK_PLATFORM "func_sim" CACHE STRING "Platform hook to use (func_sim)")
set(STUB_BUILD_TARGET "all" CACHE STRING "Build target: app, ut, or all")

# ============================================================
# Compiler settings
# ============================================================
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Werror)

# ============================================================
# Platform hooks
# ============================================================
add_subdirectory(hooks/${HOOK_PLATFORM}/lib)

# ============================================================
# Common library (registry + config + platform wrappers)
# ============================================================
add_subdirectory(common)
# stub_common needs platform hook headers for platform_api.h
target_include_directories(stub_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/hooks/${HOOK_PLATFORM}/include
)

# ============================================================
# Models — auto-generates stub_model_table.c
# ============================================================
add_subdirectory(models)

# ============================================================
# App target: stub_runner executable
# ============================================================
if(STUB_BUILD_TARGET STREQUAL "app" OR STUB_BUILD_TARGET STREQUAL "all")
    add_executable(stub_runner
        hooks/${HOOK_PLATFORM}/src/${HOOK_PLATFORM}_main.c
        common/src/stub_main.c
        ${GENERATED_MODEL_TABLE}
        ${ALL_MODEL_SOURCES}
    )
    target_link_libraries(stub_runner PRIVATE
        stub_common
        platform_sim
    )
endif()

# ============================================================
# UT target
# ============================================================
if(STUB_BUILD_TARGET STREQUAL "ut" OR STUB_BUILD_TARGET STREQUAL "all")
    enable_testing()
    add_subdirectory(tests)
endif()
