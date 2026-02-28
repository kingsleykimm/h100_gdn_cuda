cmake_minimum_required(VERSION 3.22)
project(h100_gdn_cuda LANGUAGES CXX CUDA)

# ============================================================================
# Options
# ============================================================================
option(BUILD_TESTS   "Build test executables"      ON)
option(BUILD_PYBIND  "Build Python extension"      ON)
option(FETCH_LIBTORCH "Auto-fetch LibTorch when Torch is not found" OFF)
set(LIBTORCH_VERSION "2.10.0" CACHE STRING "LibTorch version to fetch")
set(LIBTORCH_CUDA_TAG "" CACHE STRING "Override CUDA tag (e.g. cu128, cu126, cu124, cu118)")

# ============================================================================
# C++ / CUDA standards
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90a")
# Match qwen2.c layout: place built artifacts directly in the chosen build dir
# (e.g., `build/` or `build_cmake/`) instead of a nested `bin/`.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR})
endforeach()

# ============================================================================
# Find packages
# ============================================================================
include(FetchContent)
# Keep FetchContent downloads outside build/ so clean builds don't re-download.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake_deps" CACHE PATH "Persistent FetchContent cache")

find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "9.0a" CACHE STRING "Torch CUDA arch list")
endif()
message(STATUS "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")

# Prefer Torch from the active Python environment (similar to DeepGEMM flow)
# before falling back to downloading LibTorch.
if(BUILD_PYBIND AND NOT DEFINED Torch_DIR)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TORCH_CMAKE_PATH)
            message(STATUS "Using PyTorch from Python environment: ${TORCH_CMAKE_PATH}")
            list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
        endif()
    endif()
endif()

find_package(Torch QUIET CONFIG)

if(NOT Torch_FOUND AND FETCH_LIBTORCH AND NOT DEFINED Torch_DIR)
    if(LIBTORCH_CUDA_TAG)
        set(_libtorch_cuda_tag "${LIBTORCH_CUDA_TAG}")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR GREATER_EQUAL 8)
        set(_libtorch_cuda_tag "cu128")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR EQUAL 6)
        set(_libtorch_cuda_tag "cu126")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR EQUAL 4)
        set(_libtorch_cuda_tag "cu124")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR EQUAL 8)
        set(_libtorch_cuda_tag "cu118")
    else()
        message(FATAL_ERROR
            "No preconfigured LibTorch mapping for CUDA ${CUDAToolkit_VERSION}. "
            "Set -DLIBTORCH_CUDA_TAG=<cuXXX> manually (e.g. cu128).")
    endif()

    set(_libtorch_archive "libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${_libtorch_cuda_tag}.zip")
    set(_libtorch_url "https://download.pytorch.org/libtorch/${_libtorch_cuda_tag}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${_libtorch_cuda_tag}.zip")
    message(STATUS "Fetching LibTorch ${LIBTORCH_VERSION}+${_libtorch_cuda_tag} from: ${_libtorch_url}")

    FetchContent_Declare(
        libtorch_prebuilt
        URL "${_libtorch_url}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(libtorch_prebuilt)
    if(NOT libtorch_prebuilt_POPULATED)
        FetchContent_Populate(libtorch_prebuilt)
    endif()

    set(Torch_DIR "${libtorch_prebuilt_SOURCE_DIR}/share/cmake/Torch" CACHE PATH "Torch CMake config path" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${libtorch_prebuilt_SOURCE_DIR}")
    message(STATUS "Using fetched LibTorch from ${libtorch_prebuilt_SOURCE_DIR}")
    find_package(Torch REQUIRED CONFIG)
elseif(NOT Torch_FOUND)
    message(FATAL_ERROR
        "Torch was not found. Either install PyTorch in this Python env, set -DTorch_DIR, "
        "or configure with -DFETCH_LIBTORCH=ON to download LibTorch.")
endif()

# ============================================================================
# Third-party: CUTLASS (headers only)
# ============================================================================
set(SUBMODULE_CUTLASS_INCLUDE "${CMAKE_SOURCE_DIR}/third-party/cutlass/include")
if(EXISTS "${SUBMODULE_CUTLASS_INCLUDE}/cutlass" AND EXISTS "${SUBMODULE_CUTLASS_INCLUDE}/cute")
    set(CUTLASS_INCLUDE "${SUBMODULE_CUTLASS_INCLUDE}")
    message(STATUS "Using CUTLASS submodule headers from ${CUTLASS_INCLUDE}")
else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG        v3.8.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(cutlass)
    if(NOT cutlass_POPULATED)
        FetchContent_Populate(cutlass)
    endif()
    set(CUTLASS_INCLUDE ${cutlass_SOURCE_DIR}/include)
    message(STATUS "Using FetchContent CUTLASS headers from ${CUTLASS_INCLUDE}")
endif()

# ============================================================================
# Third-party: fmt
# ============================================================================
set(SUBMODULE_FMT_DIR "${CMAKE_SOURCE_DIR}/third-party/fmt")
if(EXISTS "${SUBMODULE_FMT_DIR}/CMakeLists.txt")
    add_subdirectory("${SUBMODULE_FMT_DIR}" "${CMAKE_BINARY_DIR}/third_party_fmt" EXCLUDE_FROM_ALL)
    message(STATUS "Using fmt submodule from ${SUBMODULE_FMT_DIR}")
else()
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        10.2.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(fmt)
    message(STATUS "Using FetchContent fmt")
endif()

# ============================================================================
# Use pre-fetched CUTLASS / CuTe headers if they exist in include/
# ============================================================================
set(LOCAL_CUTLASS_INCLUDE "${CMAKE_SOURCE_DIR}/include/cutlass")
set(LOCAL_CUTE_INCLUDE    "${CMAKE_SOURCE_DIR}/include/cute")
if(EXISTS "${LOCAL_CUTLASS_INCLUDE}" AND EXISTS "${LOCAL_CUTE_INCLUDE}")
    set(CUTLASS_INCLUDE "${CMAKE_SOURCE_DIR}/include")
    message(STATUS "Using local CUTLASS headers from ${CUTLASS_INCLUDE}")
else()
    message(STATUS "Using CUTLASS headers from ${CUTLASS_INCLUDE}")
endif()

# ============================================================================
# Core library (gdn_cuda_lib)
# ============================================================================
add_library(gdn_cuda_lib STATIC
    src/gdn_ops.cu
    csrc/apis/gdn_forward.cpp
)

target_include_directories(gdn_cuda_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/gdn_cuda/include
        ${CUTLASS_INCLUDE}
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/csrc
)

target_link_libraries(gdn_cuda_lib PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    fmt::fmt
)
target_compile_definitions(gdn_cuda_lib PUBLIC
    # fmt v12 can trigger NVCC errors with uint128 codepaths; disable int128 usage.
    FMT_USE_INT128=0
)

set_target_properties(gdn_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# NVCC flags
target_compile_options(gdn_cuda_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
        --use_fast_math
        -lineinfo
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fPIC
    >
)

# ============================================================================
# Python extension (optional)
# ============================================================================
if(BUILD_PYBIND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(gdn_cuda csrc/python_api.cpp)
    target_include_directories(gdn_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/gdn_cuda/include
        ${CMAKE_SOURCE_DIR}/csrc
        ${CUTLASS_INCLUDE}
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(gdn_cuda PRIVATE
        gdn_cuda_lib
        ${TORCH_LIBRARIES}
        fmt::fmt
    )
    # Link torch_python explicitly when available so at::Tensor pybind casters resolve.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os, torch; p=os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'); print(p if os.path.exists(p) else '')"
        OUTPUT_VARIABLE TORCH_PYTHON_LIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_PYTHON_LIB)
        message(STATUS "Linking torch_python from ${TORCH_PYTHON_LIB}")
        target_link_libraries(gdn_cuda PRIVATE "${TORCH_PYTHON_LIB}")
    else()
        message(WARNING "libtorch_python.so not found in Python torch package; proceeding without explicit torch_python link.")
    endif()
    target_compile_definitions(gdn_cuda PRIVATE
        FMT_USE_INT128=0
    )
    target_compile_options(gdn_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    )
    target_compile_definitions(gdn_cuda PRIVATE
        TORCH_EXTENSION_NAME=gdn_cuda
    )
endif()

# ============================================================================
# Test executables (optional)
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    set(TEST_SOURCES
        test/test_cpp/test_gdn_helpers.cpp
        test/test_cpp/test_gdn_compute_u_w.cpp
        test/test_cpp/test_gdn_seq_state_passing.cpp
        test/test_cpp/test_gdn_chunked_compute_O.cpp
        test/test_cpp/test_gdn_fwd.cpp
        test/test_cpp/test_gdn_recurrent.cpp
    )

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        if(TEST_NAME STREQUAL "test_gdn_helpers")
            set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE CUDA)
        endif()
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/gdn_cuda/include
            ${CMAKE_SOURCE_DIR}/csrc
            ${CMAKE_SOURCE_DIR}/test
            ${CUTLASS_INCLUDE}
            ${TORCH_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${TEST_NAME} PRIVATE
            gdn_cuda_lib
            ${TORCH_LIBRARIES}
            CUDA::cudart
            fmt::fmt
        )
        target_compile_definitions(${TEST_NAME} PRIVATE
            FMT_USE_INT128=0
        )
        target_compile_options(${TEST_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-relaxed-constexpr --expt-extended-lambda>
        )
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO}
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL}
        )
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS gdn_cuda_lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY gdn_cuda/include/gdn_cuda DESTINATION include)
