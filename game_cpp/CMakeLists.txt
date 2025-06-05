cmake_minimum_required(VERSION 3.17)
project(azul_openspiel_local)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# Suppress warnings for OpenSpiel includes by treating them as system includes
include_directories(SYSTEM ${OPENSPIEL_INCLUDE_DIRS})

# Add compiler flags to suppress OpenSpiel warnings
if(NOT MSVC)
    # Add flags to suppress common OpenSpiel warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare -Wno-reorder")
    # Optionally suppress all warnings for third-party code
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-but-set-variable")
endif()

# Create local Azul game library
add_library(azul_local STATIC
    azul.cc
    azul.h
)

# OpenSpiel includes are now handled as system includes globally

# Agent evaluation library
add_library(azul_evaluation STATIC
    evaluation_config.cpp
    agent_evaluator.cpp
    agent_profiler.cpp
    mcts_agent.cpp
    random_agent.cpp
    minimax_agent.cpp
)

target_include_directories(azul_evaluation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Evaluation demo executable (comprehensive)
add_executable(azul_evaluation_demo
    main_evaluation_demo.cpp
)

target_include_directories(azul_evaluation_demo PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(azul_evaluation_demo
    azul_evaluation
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
)

# Status messages
message(STATUS "")
message(STATUS "✅ Simplified OpenSpiel Azul integration configured")
message(STATUS "📁 OpenSpiel source: ${OPENSPIEL_ROOT}")
message(STATUS "📚 Shared library: ${LOCAL_OPENSPIEL_LIB}")
message(STATUS "🎯 Build targets: azul_evaluation_demo")
message(STATUS "")
message(STATUS "🚀 To build and run:")
message(STATUS "   make azul_evaluation_demo && ./azul_evaluation_demo")
message(STATUS "")
