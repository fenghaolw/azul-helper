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

# OpenSpiel dependency (local approach)
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
set(OPENSPIEL_ROOT "${GRANDPARENT_DIR}/open_spiel")

# Check if OpenSpiel source exists (for headers)
if(EXISTS "${OPENSPIEL_ROOT}" AND EXISTS "${OPENSPIEL_ROOT}/open_spiel")
    message(STATUS "Found OpenSpiel source at: ${OPENSPIEL_ROOT}")
    
    # Include OpenSpiel headers from source
    include_directories(${OPENSPIEL_ROOT})
    include_directories(${OPENSPIEL_ROOT}/open_spiel)
    include_directories(${OPENSPIEL_ROOT}/open_spiel/abseil-cpp)
    
    # Use local shared library
    set(LOCAL_OPENSPIEL_LIB "${CMAKE_CURRENT_SOURCE_DIR}/libopen_spiel.dylib")
    if(EXISTS "${LOCAL_OPENSPIEL_LIB}")
        message(STATUS "Found local OpenSpiel shared library: ${LOCAL_OPENSPIEL_LIB}")
        set(WITH_OPENSPIEL TRUE)
    else()
        message(FATAL_ERROR "Local OpenSpiel shared library not found: ${LOCAL_OPENSPIEL_LIB}")
    endif()
    
    # Find Abseil libraries (required by OpenSpiel)
    file(GLOB_RECURSE ABSEIL_LIBRARIES "${OPENSPIEL_ROOT}/build/abseil-cpp/absl/*/*.a")
    if(ABSEIL_LIBRARIES)
        list(LENGTH ABSEIL_LIBRARIES ABSEIL_COUNT)
        message(STATUS "Found ${ABSEIL_COUNT} Abseil libraries")
    else()
        message(WARNING "No Abseil libraries found - may need to build OpenSpiel first")
        set(ABSEIL_LIBRARIES "")
    endif()
    
else()
    message(FATAL_ERROR "OpenSpiel source not found at expected location: ${OPENSPIEL_ROOT}")
endif()

# Define OPENSPIEL integration
add_definitions(-DWITH_OPENSPIEL)

# Create a local Azul game library from our forked files
add_library(azul_local STATIC
    azul.cc
    azul.h
)

target_include_directories(azul_local PRIVATE
    ${OPENSPIEL_ROOT}
    ${OPENSPIEL_ROOT}/open_spiel
    ${OPENSPIEL_ROOT}/open_spiel/abseil-cpp
)

# Simple MCTS demo using local Azul game
add_executable(azul_mcts_demo
    test_mcts_integration.cpp
)

target_include_directories(azul_mcts_demo PRIVATE
    ${OPENSPIEL_ROOT}
    ${OPENSPIEL_ROOT}/open_spiel
    ${OPENSPIEL_ROOT}/open_spiel/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(azul_mcts_demo
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
    ${ABSEIL_LIBRARIES}
)

# Test program for updated agents using local Azul game
add_executable(azul_agents_test
    random_agent.cpp
    minimax_agent.cpp
)

target_include_directories(azul_agents_test PRIVATE
    ${OPENSPIEL_ROOT}
    ${OPENSPIEL_ROOT}/open_spiel
    ${OPENSPIEL_ROOT}/open_spiel/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(azul_agents_test
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
    ${ABSEIL_LIBRARIES}
)

message(STATUS "")
message(STATUS "✅ Local OpenSpiel Azul integration configured")
message(STATUS "📁 OpenSpiel source: ${OPENSPIEL_ROOT}")
message(STATUS "📚 Local shared library: ${LOCAL_OPENSPIEL_LIB}")
message(STATUS "🎯 Local Azul files: azul.cc, azul.h")
message(STATUS "🎯 Build targets:")
message(STATUS "   • azul_mcts_demo (Local Azul with OpenSpiel MCTS)")
message(STATUS "   • azul_agents_test (Local Azul with updated agents)")
message(STATUS "")
message(STATUS "🚀 To build and run:")
message(STATUS "   make azul_mcts_demo && ./azul_mcts_demo")
message(STATUS "   make azul_agents_test && ./azul_agents_test")
message(STATUS "")
