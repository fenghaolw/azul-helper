cmake_minimum_required(VERSION 3.12)
project(azul_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect and prefer current Python executable (respects virtual environment)
if(DEFINED ENV{VIRTUAL_ENV})
    message(STATUS "Virtual environment detected: $ENV{VIRTUAL_ENV}")
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()

# Find Python - prefer the virtual environment version
find_package(Python3 COMPONENTS Interpreter Development)

message(STATUS "Python3_FOUND: ${Python3_FOUND}")
if(Python3_FOUND)
    message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
    message(STATUS "Python3_VERSION: ${Python3_VERSION}")
endif()

# Try to find pybind11 from pip installation
if(Python3_FOUND)
    message(STATUS "Searching for pybind11 using Python: ${Python3_EXECUTABLE}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE pybind11_ERROR
        RESULT_VARIABLE pybind11_RESULT
    )
    
    message(STATUS "pybind11 command result: ${pybind11_RESULT}")
    if(pybind11_DIR AND pybind11_RESULT EQUAL 0)
        message(STATUS "Found pybind11 cmake dir: ${pybind11_DIR}")
        find_package(pybind11)
        if(pybind11_FOUND)
            message(STATUS "Successfully found pybind11 from pip: ${pybind11_DIR}")
        endif()
    else()
        message(STATUS "Failed to get pybind11 cmake dir from pip")
        if(pybind11_ERROR)
            message(STATUS "Error: ${pybind11_ERROR}")
        endif()
    endif()
endif()

# If pip version not found, try system installation
if(NOT pybind11_FOUND)
    find_package(pybind11)
    if(pybind11_FOUND)
        message(STATUS "Found pybind11 from system installation")
    endif()
endif()

# OpenSpiel dependency (optional - set OPENSPIEL_ROOT if available)
set(OPENSPIEL_ROOT "" CACHE PATH "Path to OpenSpiel installation")
if(OPENSPIEL_ROOT)
    find_library(OPENSPIEL_LIB openspiel PATHS ${OPENSPIEL_ROOT}/build)
    if(OPENSPIEL_LIB)
        add_definitions(-DWITH_OPENSPIEL)
        include_directories(${OPENSPIEL_ROOT})
        set(OPENSPIEL_LIBRARIES ${OPENSPIEL_LIB})
    endif()
endif()

# Core Azul game library
set(AZUL_SOURCES
    tile.cpp
    action.cpp
    player_board.cpp
    factory.cpp
    game_state.cpp
)

add_library(azul_core ${AZUL_SOURCES})

# OpenSpiel integration (if available)
if(OPENSPIEL_LIBRARIES)
    add_library(azul_openspiel azul_openspiel.cpp)
    target_link_libraries(azul_openspiel azul_core ${OPENSPIEL_LIBRARIES})
    
    # MCTS Agent
    add_library(azul_mcts mcts_agent.cpp)
    target_link_libraries(azul_mcts azul_core azul_openspiel ${OPENSPIEL_LIBRARIES})
endif()

# Python bindings (only if pybind11 is available)
if(pybind11_FOUND)
    message(STATUS "Building Python bindings with pybind11")
    pybind11_add_module(azul_cpp_bindings python_bindings.cpp)
    target_link_libraries(azul_cpp_bindings PRIVATE azul_core)

    if(OPENSPIEL_LIBRARIES)
        target_link_libraries(azul_cpp_bindings PRIVATE azul_openspiel azul_mcts)
    endif()
    
    # Export compile commands for IDE support
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
else()
    message(STATUS "pybind11 not found - Python bindings will not be built")
endif()

# Compiler-specific options
target_compile_features(azul_core PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(azul_core PRIVATE /W4)
else()
    target_compile_options(azul_core PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

# Comprehensive game engine tests
add_executable(test_basic_functionality test_basic_functionality.cpp)
target_link_libraries(test_basic_functionality azul_core)
target_compile_features(test_basic_functionality PRIVATE cxx_std_17)

add_executable(test_legal_moves test_legal_moves.cpp)
target_link_libraries(test_legal_moves azul_core)
target_compile_features(test_legal_moves PRIVATE cxx_std_17)

add_executable(test_game_flow test_game_flow.cpp)
target_link_libraries(test_game_flow azul_core)
target_compile_features(test_game_flow PRIVATE cxx_std_17)

add_executable(test_scoring test_scoring.cpp)
target_link_libraries(test_scoring azul_core)
target_compile_features(test_scoring PRIVATE cxx_std_17)

add_executable(test_edge_cases test_edge_cases.cpp)
target_link_libraries(test_edge_cases azul_core)
target_compile_features(test_edge_cases PRIVATE cxx_std_17)

# Test runner (shell script approach is better for multiple executables)