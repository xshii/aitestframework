# ============================================================
# Platform sim mock (test-only)
# ============================================================
add_library(platform_sim STATIC platform_sim.c)
target_include_directories(platform_sim PUBLIC
    ${CMAKE_SOURCE_DIR}/hooks/${HOOK_PLATFORM}/include
)

# ============================================================
# Unit tests
# ============================================================
set(TEST_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/common/memmap
    ${CMAKE_SOURCE_DIR}/common/weight
    ${CMAKE_SOURCE_DIR}/common/result
    ${CMAKE_SOURCE_DIR}/hooks/${HOOK_PLATFORM}/include
)

# --- test_registry ---
add_executable(test_registry test_registry.c)
target_include_directories(test_registry PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_registry PRIVATE stub_common)
add_test(NAME test_registry COMMAND test_registry)

# --- test_platform ---
add_executable(test_platform test_platform.c)
target_include_directories(test_platform PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_platform PRIVATE stub_common platform_sim)
add_test(NAME test_platform COMMAND test_platform)

# --- test_tdd ---
add_executable(test_tdd
    test_tdd.c
    ${ALL_MODEL_SOURCES}
    ${GENERATED_MODEL_TABLE}
)
target_include_directories(test_tdd PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_tdd PRIVATE stub_common platform_sim)
add_test(NAME test_tdd COMMAND test_tdd)

# --- test_fdd ---
add_executable(test_fdd
    test_fdd.c
    ${ALL_MODEL_SOURCES}
    ${GENERATED_MODEL_TABLE}
)
target_include_directories(test_fdd PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_fdd PRIVATE stub_common platform_sim)
add_test(NAME test_fdd COMMAND test_fdd)

# --- test_memmap ---
add_executable(test_memmap test_memmap.c)
target_include_directories(test_memmap PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_memmap PRIVATE stub_common platform_sim)
add_test(NAME test_memmap COMMAND test_memmap)

# --- test_weight ---
add_executable(test_weight test_weight.c)
target_include_directories(test_weight PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_weight PRIVATE stub_common platform_sim)
add_test(NAME test_weight COMMAND test_weight)

# --- test_result ---
add_executable(test_result test_result.c)
target_include_directories(test_result PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_result PRIVATE stub_common platform_sim)
add_test(NAME test_result COMMAND test_result)

# --- func_sim_main (integration test runner) ---
add_executable(func_sim_main
    func_sim_main.c
    platform_sim.c
    ${ALL_MODEL_SOURCES}
    ${GENERATED_MODEL_TABLE}
    ${CMAKE_SOURCE_DIR}/common/src/stub_main.c
)
target_include_directories(func_sim_main PRIVATE ${TEST_INCLUDES})
target_link_libraries(func_sim_main PRIVATE stub_common)
