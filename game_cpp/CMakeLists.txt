cmake_minimum_required(VERSION 3.12)
project(azul_cpp)
enable_testing()

# Set default C++ flags and build type specific flags
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

if(MSVC)
    set(DEFAULT_CXX_WARNING_FLAGS "/W4")
else()
    set(DEFAULT_CXX_WARNING_FLAGS "-Wall -Wextra -Wpedantic")
endif()

# Append general warning flags to CMAKE_CXX_FLAGS if it's already set by user/toolchain, otherwise set them.
if(NOT CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_WARNING_FLAGS}")
else()
    string(APPEND CMAKE_CXX_FLAGS " ${DEFAULT_CXX_WARNING_FLAGS}")
endif()

# Debug flags
if(NOT CMAKE_CXX_FLAGS_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif()

# Release flags
if(NOT CMAKE_CXX_FLAGS_RELEASE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Release with Debug Info flags
if(NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
endif()

# Minimum Size Release flags
if(NOT CMAKE_CXX_FLAGS_MINSIZEREL)
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
endif()

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

# Try to find pybind11
if(Python3_FOUND)
    find_package(pybind11 REQUIRED)
    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_DIR}")
    # No specific message needed here if not found,
    # find_package with REQUIRED will show an error.
    # The existing fallback message at the end of the file will also be printed.
    endif()
# The find_package(pybind11 REQUIRED) call above will handle the case where pybind11 is not found.
# The message "pybind11 not found - Python bindings will not be built"
# later in the script will still be displayed if pybind11_FOUND is FALSE.
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

# Compiler-specific options for azul_core
target_compile_features(azul_core PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(azul_core PRIVATE /W4) # MSVC specific warnings
else()
    target_compile_options(azul_core PRIVATE -Wall -Wextra -Wpedantic) # GCC/Clang specific warnings
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

# Apply warning flags to all test executables
set(AZUL_TEST_EXECUTABLES
    test_basic_functionality
    test_legal_moves
    test_game_flow
    test_scoring
    test_edge_cases
)

foreach(test_exe ${AZUL_TEST_EXECUTABLES})
    if(MSVC)
        target_compile_options(${test_exe} PRIVATE /W4)
    else()
        target_compile_options(${test_exe} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

# Add tests to CTest
add_test(NAME test_basic_functionality COMMAND test_basic_functionality)
add_test(NAME test_legal_moves COMMAND test_legal_moves)
add_test(NAME test_game_flow COMMAND test_game_flow)
add_test(NAME test_scoring COMMAND test_scoring)
add_test(NAME test_edge_cases COMMAND test_edge_cases)
