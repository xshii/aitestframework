add_library(stub_common STATIC
    src/stub_registry.c
    src/stub_config.c
    src/stub_platform.c
)

target_include_directories(stub_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
