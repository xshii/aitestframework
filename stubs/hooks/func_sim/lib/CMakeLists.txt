add_library(platform_sim STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/platform_sim.c
)

target_include_directories(platform_sim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
