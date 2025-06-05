cmake_minimum_required(VERSION 3.17)
project(azul_openspiel_local)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "/W4")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# OpenSpiel paths
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
set(OPENSPIEL_ROOT "${GRANDPARENT_DIR}/open_spiel")
set(LOCAL_OPENSPIEL_LIB "${CMAKE_CURRENT_SOURCE_DIR}/libopen_spiel.dylib")

# Verify required files exist
if(NOT EXISTS "${OPENSPIEL_ROOT}/open_spiel")
    message(FATAL_ERROR "OpenSpiel source not found at: ${OPENSPIEL_ROOT}")
endif()

if(NOT EXISTS "${LOCAL_OPENSPIEL_LIB}")
    message(FATAL_ERROR "Local OpenSpiel shared library not found: ${LOCAL_OPENSPIEL_LIB}")
endif()

# Include directories (consolidated)
set(OPENSPIEL_INCLUDE_DIRS
    ${OPENSPIEL_ROOT}
    ${OPENSPIEL_ROOT}/open_spiel
    ${OPENSPIEL_ROOT}/open_spiel/abseil-cpp
)



# Create local Azul game library
add_library(azul_local STATIC
    azul.cc
    azul.h
)

target_include_directories(azul_local PRIVATE ${OPENSPIEL_INCLUDE_DIRS})

# MCTS demo executable
add_executable(azul_mcts_demo
    test_mcts_integration.cpp
)

target_include_directories(azul_mcts_demo PRIVATE 
    ${OPENSPIEL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(azul_mcts_demo
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
)

# Agents test executable
add_executable(azul_agents_test
    random_agent.cpp
    minimax_agent.cpp
)

target_include_directories(azul_agents_test PRIVATE 
    ${OPENSPIEL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(azul_agents_test
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
)

# Status messages
message(STATUS "")
message(STATUS "✅ Simplified OpenSpiel Azul integration configured")
message(STATUS "📁 OpenSpiel source: ${OPENSPIEL_ROOT}")
message(STATUS "📚 Shared library: ${LOCAL_OPENSPIEL_LIB}")
message(STATUS "🎯 Build targets: azul_mcts_demo, azul_agents_test")
message(STATUS "")
message(STATUS "🚀 To build and run:")
message(STATUS "   make azul_mcts_demo && ./azul_mcts_demo")
message(STATUS "   make azul_agents_test && ./azul_agents_test")
message(STATUS "")
