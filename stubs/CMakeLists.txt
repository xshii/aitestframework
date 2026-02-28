cmake_minimum_required(VERSION 3.12)
project(stub_framework C)

# ============================================================
# Options
# ============================================================
set(HOOK_PLATFORM "func_sim" CACHE STRING "Hook platform (func_sim)")
set(STUB_BUILD_TARGET "all" CACHE STRING "Build target: app | ut | all")
option(EMBED_WEIGHTS "Embed weight binaries into the compiled binary" OFF)
set(EMBED_WEIGHT_MANIFEST "" CACHE FILEPATH "Weight manifest for embedding")
set(EMBED_WEIGHT_DIR "" CACHE PATH "Base directory for weight bin files")

# ============================================================
# Compiler settings
# ============================================================
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Werror)

# ============================================================
# Platform hook headers
# ============================================================
set(HOOK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hooks/${HOOK_PLATFORM}/include)

# ============================================================
# Common library
# ============================================================
add_subdirectory(common)
target_include_directories(stub_common PUBLIC ${HOOK_INCLUDE_DIR})

# ============================================================
# Models — auto-generates stub_model_table.c
# ============================================================
add_subdirectory(models)

# ============================================================
# App target: stub_runner.so
# ============================================================
if(STUB_BUILD_TARGET STREQUAL "app" OR STUB_BUILD_TARGET STREQUAL "all")
    add_library(stub_runner SHARED
        common/src/stub_main.c
        ${GENERATED_MODEL_TABLE}
        ${ALL_MODEL_SOURCES}
    )
    target_link_libraries(stub_runner PRIVATE stub_common)
endif()

# ============================================================
# UT target
# ============================================================
if(STUB_BUILD_TARGET STREQUAL "ut" OR STUB_BUILD_TARGET STREQUAL "all")
    enable_testing()
    add_subdirectory(tests)
endif()
