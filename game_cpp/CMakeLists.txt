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

# Find LibTorch from OpenSpiel installation
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
set(OPENSPIEL_ROOT "${GRANDPARENT_DIR}/open_spiel")
set(LOCAL_OPENSPIEL_LIB "${CMAKE_CURRENT_SOURCE_DIR}/libopen_spiel.dylib")

# LibTorch paths (from OpenSpiel installation)
set(Torch_DIR "${OPENSPIEL_ROOT}/open_spiel/libtorch/libtorch/share/cmake/Torch")
unset(CMAKE_PREFIX_PATH CACHE)
set(CMAKE_PREFIX_PATH "${OPENSPIEL_ROOT}/open_spiel/libtorch/libtorch")
find_package(Torch REQUIRED PATHS "${OPENSPIEL_ROOT}/open_spiel/libtorch/libtorch" NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Verify required files exist
if(NOT EXISTS "${OPENSPIEL_ROOT}/open_spiel")
    message(FATAL_ERROR "OpenSpiel source not found at: ${OPENSPIEL_ROOT}")
endif()

if(NOT EXISTS "${LOCAL_OPENSPIEL_LIB}")
    message(FATAL_ERROR "Local OpenSpiel shared library not found: ${LOCAL_OPENSPIEL_LIB}")
endif()

# Find nlohmann-json
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(NLOHMANN_JSON QUIET nlohmann_json)
endif()

# Fallback to find nlohmann-json in Homebrew
if(NOT NLOHMANN_JSON_FOUND)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS 
            /opt/homebrew/include
            /usr/local/include
            ${CMAKE_PREFIX_PATH}/include
        NO_DEFAULT_PATH
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        set(NLOHMANN_JSON_FOUND TRUE)
        message(STATUS "Found nlohmann-json: ${NLOHMANN_JSON_INCLUDE_DIR}")
    endif()
endif()

if(NOT NLOHMANN_JSON_FOUND)
    message(FATAL_ERROR "nlohmann-json not found. Install with: brew install nlohmann-json")
endif()

# Find cxxopts
find_path(CXXOPTS_INCLUDE_DIR cxxopts.hpp
    PATHS 
        /opt/homebrew/include
        /usr/local/include
        ${CMAKE_PREFIX_PATH}/include
    NO_DEFAULT_PATH
)

if(CXXOPTS_INCLUDE_DIR)
    message(STATUS "Found cxxopts: ${CXXOPTS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "cxxopts not found. Install with: brew install cxxopts")
endif()

# Find cpp-httplib
find_path(CPP_HTTPLIB_INCLUDE_DIR httplib.h
    PATHS 
        /opt/homebrew/include
        /usr/local/include
        ${CMAKE_PREFIX_PATH}/include
    NO_DEFAULT_PATH
)

if(CPP_HTTPLIB_INCLUDE_DIR)
    message(STATUS "Found cpp-httplib: ${CPP_HTTPLIB_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "cpp-httplib not found. Install with: brew install cpp-httplib")
endif()

# Include directories (consolidated)
set(OPENSPIEL_INCLUDE_DIRS
    ${OPENSPIEL_ROOT}
    ${OPENSPIEL_ROOT}/open_spiel
    ${OPENSPIEL_ROOT}/open_spiel/abseil-cpp
)

# Suppress warnings for OpenSpiel includes by treating them as system includes
include_directories(SYSTEM ${OPENSPIEL_INCLUDE_DIRS})

# Add nlohmann-json include directory
if(NLOHMANN_JSON_INCLUDE_DIR)
    include_directories(SYSTEM ${NLOHMANN_JSON_INCLUDE_DIR})
else()
    include_directories(SYSTEM ${NLOHMANN_JSON_INCLUDE_DIRS})
endif()

# Add cxxopts include directory
include_directories(SYSTEM ${CXXOPTS_INCLUDE_DIR})

# Add cpp-httplib include directory
include_directories(SYSTEM ${CPP_HTTPLIB_INCLUDE_DIR})

# Add compiler flags to suppress OpenSpiel warnings
if(NOT MSVC)
    # Add flags to suppress common OpenSpiel warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare -Wno-reorder")
    # Optionally suppress all warnings for third-party code
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-but-set-variable")
endif()

# Find nlohmann_json package
find_package(nlohmann_json 3.11.2 REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

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
    ${ALPHA_ZERO_TORCH_DIR}
    ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
)

# Link with LibTorch if available for AlphaZero support
target_link_libraries(azul_evaluation
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Evaluation demo executable (comprehensive)
add_executable(azul_evaluation_demo
    main_evaluation_demo.cpp
)

target_include_directories(azul_evaluation_demo PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALPHA_ZERO_TORCH_DIR}
    ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
)

target_link_libraries(azul_evaluation_demo
    azul_evaluation
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
    alpha_zero_torch
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Profiling demo executable
add_executable(azul_profiling_demo
    main_profiling_demo.cpp
)

target_include_directories(azul_profiling_demo PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALPHA_ZERO_TORCH_DIR}
    ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
)

target_link_libraries(azul_profiling_demo
    azul_evaluation
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
    alpha_zero_torch
    ${TORCH_LIBRARIES}
)

# Neural MCTS Training executable with LibTorch
add_executable(neural_mcts_trainer
    neural_mcts_training.cpp
)

target_include_directories(neural_mcts_trainer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(neural_mcts_trainer
    azul_evaluation
    azul_local
    ${LOCAL_OPENSPIEL_LIB}
    ${TORCH_LIBRARIES}
)

# Copy the LibTorch shared libraries to the binary directory
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET neural_mcts_trainer
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:neural_mcts_trainer>)
endif(MSVC)

# Add LibTorch AlphaZero support
set(ALPHA_ZERO_TORCH_DIR "${OPENSPIEL_ROOT}/open_spiel/algorithms/alpha_zero_torch")

# Check if LibTorch AlphaZero sources are available
if(EXISTS "${ALPHA_ZERO_TORCH_DIR}/alpha_zero.cc")
    message(STATUS "🔥 LibTorch AlphaZero sources found")
    
    # LibTorch AlphaZero library
    add_library(alpha_zero_torch STATIC
        ${ALPHA_ZERO_TORCH_DIR}/alpha_zero.cc
        ${ALPHA_ZERO_TORCH_DIR}/model.cc
        ${ALPHA_ZERO_TORCH_DIR}/vpnet.cc
        ${ALPHA_ZERO_TORCH_DIR}/vpevaluator.cc
    )
    
    target_include_directories(alpha_zero_torch PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ALPHA_ZERO_TORCH_DIR}
        ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
    )
    
    target_link_libraries(alpha_zero_torch
        ${LOCAL_OPENSPIEL_LIB}
        ${TORCH_LIBRARIES}
    )
    
    # LibTorch AlphaZero Training executable
    add_executable(libtorch_alphazero_trainer
        libtorch_alphazero_training.cpp
    )
    
    target_include_directories(libtorch_alphazero_trainer PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ALPHA_ZERO_TORCH_DIR}
        ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
    )
    
    target_link_libraries(libtorch_alphazero_trainer
        alpha_zero_torch
        azul_local
        ${LOCAL_OPENSPIEL_LIB}
        ${TORCH_LIBRARIES}
    )
    
    # Also add alpha_zero_torch support to evaluation demo
    target_link_libraries(azul_evaluation_demo
        alpha_zero_torch
        ${TORCH_LIBRARIES}
    )
    
    message(STATUS "🎯 LibTorch AlphaZero build target: libtorch_alphazero_trainer")
    
    # AlphaZero Bridge - JSON-enabled bridge for full integration
    add_executable(alphazero_bridge
        alphazero_bridge.cpp
    )
    
    target_include_directories(alphazero_bridge PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ALPHA_ZERO_TORCH_DIR}
        ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
    )
    
    target_link_libraries(alphazero_bridge
        azul_evaluation
        azul_local
        ${LOCAL_OPENSPIEL_LIB}
        alpha_zero_torch
        ${TORCH_LIBRARIES}
    )
    
    message(STATUS "🌉 JSON AlphaZero bridge build target: alphazero_bridge")
    
    # C++ API Server - Direct HTTP server without Python wrapper
    add_executable(azul_api_server
        azul_api_server.cpp
    )
    
    target_include_directories(azul_api_server PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ALPHA_ZERO_TORCH_DIR}
        ${OPENSPIEL_ROOT}/open_spiel/libnop/libnop/include
    )
    
    target_link_libraries(azul_api_server
        azul_evaluation
        azul_local
        ${LOCAL_OPENSPIEL_LIB}
        alpha_zero_torch
        ${TORCH_LIBRARIES}
    )
    
    message(STATUS "🌐 C++ API Server build target: azul_api_server")
else()
    message(STATUS "⚠️  LibTorch AlphaZero sources not found at ${ALPHA_ZERO_TORCH_DIR}")
    message(STATUS "    Make sure OpenSpiel was built with OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON")
endif()

# Status messages
message(STATUS "")
message(STATUS "✅ OpenSpiel Azul integration configured")
message(STATUS "📁 OpenSpiel source: ${OPENSPIEL_ROOT}")
message(STATUS "📚 Shared library: ${LOCAL_OPENSPIEL_LIB}")
message(STATUS "🔥 LibTorch found: ${TORCH_INSTALL_PREFIX}")