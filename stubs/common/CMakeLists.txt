file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/memmap/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/weight/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/result/*.c
)
# stub_main.c contains stub_entry/stub_exit, built separately into .so
list(FILTER COMMON_SOURCES EXCLUDE REGEX "stub_main\\.c$")

add_library(stub_common STATIC ${COMMON_SOURCES})
target_include_directories(stub_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/memmap
    ${CMAKE_CURRENT_SOURCE_DIR}/weight
    ${CMAKE_CURRENT_SOURCE_DIR}/result
)

# --- Embed weights into binary (optional) ---
if(EMBED_WEIGHTS)
    if(NOT EMBED_WEIGHT_MANIFEST)
        message(FATAL_ERROR "EMBED_WEIGHTS=ON requires -DEMBED_WEIGHT_MANIFEST=<path>")
    endif()

    set(_embed_out "${CMAKE_BINARY_DIR}/generated/stub_weight_embed.c")

    add_custom_command(
        OUTPUT  ${_embed_out}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/generated"
        COMMAND python3
            "${CMAKE_SOURCE_DIR}/tools/embed_weights.py"
            "${EMBED_WEIGHT_MANIFEST}"
            "${EMBED_WEIGHT_DIR}"
            "${_embed_out}"
        DEPENDS ${EMBED_WEIGHT_MANIFEST}
        COMMENT "Embedding weight binaries into C source..."
    )

    target_sources(stub_common PRIVATE ${_embed_out})
    target_compile_definitions(stub_common PUBLIC EMBED_WEIGHTS)
endif()
