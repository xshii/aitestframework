# ============================================================
# Unit tests
# ============================================================

set(TEST_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common/include
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

# --- test_demo_model_a ---
add_executable(test_demo_model_a
    test_demo_model_a.c
    ${ALL_MODEL_SOURCES}
    ${GENERATED_MODEL_TABLE}
)
target_include_directories(test_demo_model_a PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_demo_model_a PRIVATE stub_common platform_sim)
add_test(NAME test_demo_model_a COMMAND test_demo_model_a)

# --- test_demo_model_b ---
add_executable(test_demo_model_b
    test_demo_model_b.c
    ${ALL_MODEL_SOURCES}
    ${GENERATED_MODEL_TABLE}
)
target_include_directories(test_demo_model_b PRIVATE ${TEST_INCLUDES})
target_link_libraries(test_demo_model_b PRIVATE stub_common platform_sim)
add_test(NAME test_demo_model_b COMMAND test_demo_model_b)
