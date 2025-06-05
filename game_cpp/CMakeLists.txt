cmake_minimum_required(VERSION 3.17)
project(azul_openspiel_direct)

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

# OpenSpiel dependency (auto-detect)
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
set(AUTO_OPENSPIEL_PATH "${GRANDPARENT_DIR}/open_spiel")

if(EXISTS "${AUTO_OPENSPIEL_PATH}" AND EXISTS "${AUTO_OPENSPIEL_PATH}/build")
    message(STATUS "Auto-detected OpenSpiel at: ${AUTO_OPENSPIEL_PATH}")
    set(OPENSPIEL_ROOT "${AUTO_OPENSPIEL_PATH}")
    set(OPENSPIEL_BUILD_DIR "${OPENSPIEL_ROOT}/build")
    
    # Include OpenSpiel headers
    include_directories(${OPENSPIEL_ROOT})
    include_directories(${OPENSPIEL_ROOT}/open_spiel)
    include_directories(${OPENSPIEL_ROOT}/open_spiel/abseil-cpp)
    
    # Find shared library
    find_library(OPENSPIEL_LIB 
        NAMES open_spiel
        PATHS ${OPENSPIEL_BUILD_DIR}
        NO_DEFAULT_PATH
    )
    
    if(OPENSPIEL_LIB)
        message(STATUS "Found OpenSpiel shared library: ${OPENSPIEL_LIB}")
        set(WITH_OPENSPIEL TRUE)
    else()
        message(FATAL_ERROR "OpenSpiel shared library not found. Please build OpenSpiel with BUILD_SHARED_LIB=ON")
    endif()
    
    # Find Abseil libraries (required by OpenSpiel)
    file(GLOB_RECURSE ABSEIL_LIBRARIES "${OPENSPIEL_BUILD_DIR}/abseil-cpp/absl/*/*.a")
    if(ABSEIL_LIBRARIES)
        list(LENGTH ABSEIL_LIBRARIES ABSEIL_COUNT)
        message(STATUS "Found ${ABSEIL_COUNT} Abseil libraries")
    else()
        message(FATAL_ERROR "No Abseil libraries found")
    endif()
    
    # Note: Azul game is already available in OpenSpiel shared library
    # No need to copy source files since we use LoadGame("azul")
    
else()
    message(FATAL_ERROR "OpenSpiel not found at expected location: ${AUTO_OPENSPIEL_PATH}")
endif()

# Define OPENSPIEL integration
add_definitions(-DWITH_OPENSPIEL)

# Simple MCTS demo using OpenSpiel directly (no separate azul_core needed)
add_executable(azul_mcts_demo
    test_mcts_integration.cpp
)

target_include_directories(azul_mcts_demo PRIVATE
    ${OPENSPIEL_ROOT}
    ${OPENSPIEL_ROOT}/open_spiel
    ${OPENSPIEL_ROOT}/open_spiel/abseil-cpp
)

target_link_libraries(azul_mcts_demo
    ${OPENSPIEL_LIB}
    ${ABSEIL_LIBRARIES}
)

message(STATUS "")
message(STATUS "✅ Direct OpenSpiel Azul integration configured")
message(STATUS "📁 OpenSpiel path: ${OPENSPIEL_ROOT}")
message(STATUS "📚 Shared library: ${OPENSPIEL_LIB}")
message(STATUS "🎯 Build targets:")
message(STATUS "   • azul_mcts_demo (Direct OpenSpiel MCTS demo)")
message(STATUS "")
message(STATUS "🚀 To build and run:")
message(STATUS "   make azul_mcts_demo && ./azul_mcts_demo")
message(STATUS "")
