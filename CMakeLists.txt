cmake_minimum_required(VERSION 3.16)
project(pcmci C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Detect compiler and platform
if(CMAKE_C_COMPILER_ID MATCHES "Intel" OR CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(USING_INTEL_COMPILER TRUE)
    message(STATUS "Detected Intel compiler: ${CMAKE_C_COMPILER_ID}")
else()
    set(USING_INTEL_COMPILER FALSE)
endif()

# Compiler flags - handle Windows vs Linux differences
if(USING_INTEL_COMPILER)
    if(WIN32)
        # Intel ICX on Windows uses /Q prefix and MSVC-style flags
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
        set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /DDEBUG")
        set(CMAKE_C_FLAGS_RELEASE "/O3 /QxHost /DNDEBUG")
        # Note: /Qipo for interprocedural optimization (optional, slower build)
    else()
        # Intel ICX on Linux uses - prefix  
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
        set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -xHost -ipo -DNDEBUG")
    endif()
else()
    if(WIN32)
        # MSVC or clang-cl
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
        set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /DDEBUG")
        set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
    else()
        # GCC/Clang on Linux
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
        set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    endif()
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find BLAS/LAPACK - try MKL first, fall back to OpenBLAS
find_package(PkgConfig QUIET)

set(USE_MKL FALSE)
set(USE_OPENBLAS FALSE)
set(BLAS_COMPILE_FLAGS "")
set(BLAS_LINK_FLAGS "")

# Try MKL first (preferred with Intel compiler)
if(PkgConfig_FOUND AND NOT USING_INTEL_COMPILER AND NOT WIN32)
    pkg_check_modules(MKL QUIET mkl-dynamic-lp64-iomp)
endif()

if(NOT MKL_FOUND)
    set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "MKL root directory")
    if(NOT MKL_ROOT)
        if(WIN32)
            set(MKL_ROOT "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
        else()
            set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
        endif()
    endif()
    
    if(EXISTS "${MKL_ROOT}/include/mkl.h")
        set(MKL_INCLUDE_DIRS "${MKL_ROOT}/include")
        
        if(WIN32)
            set(MKL_LIBRARY_DIRS "${MKL_ROOT}/lib")
        else()
            set(MKL_LIBRARY_DIRS "${MKL_ROOT}/lib/intel64")
        endif()
        
        if(USING_INTEL_COMPILER)
            if(WIN32)
                # Windows ICX: use /Qmkl flag
                set(MKL_COMPILE_FLAGS "/Qmkl:parallel")
                set(MKL_LINK_FLAGS "/Qmkl:parallel")
                set(MKL_LIBRARIES "")
            else()
                # Linux ICX: use -qmkl flag
                set(MKL_COMPILE_FLAGS "-qmkl=parallel")
                set(MKL_LINK_FLAGS "-qmkl=parallel")
                set(MKL_LIBRARIES "")
            endif()
        else()
            # GCC/Clang/MSVC: explicit linking
            if(WIN32)
                set(MKL_LIBRARIES 
                    "${MKL_LIBRARY_DIRS}/mkl_intel_lp64.lib"
                    "${MKL_LIBRARY_DIRS}/mkl_intel_thread.lib"
                    "${MKL_LIBRARY_DIRS}/mkl_core.lib"
                )
                set(MKL_COMPILE_FLAGS "")
                set(MKL_LINK_FLAGS "")
            else()
                set(MKL_LIBRARIES 
                    "-L${MKL_LIBRARY_DIRS}"
                    "-Wl,--no-as-needed"
                    "-lmkl_intel_lp64"
                    "-lmkl_gnu_thread"  
                    "-lmkl_core"
                    "-lgomp"
                    "-lpthread"
                    "-lm"
                    "-ldl"
                )
                set(MKL_COMPILE_FLAGS "")
                set(MKL_LINK_FLAGS "")
            endif()
        endif()
        set(MKL_FOUND TRUE)
    endif()
endif()

if(MKL_FOUND)
    set(USE_MKL TRUE)
    set(BLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLAS_COMPILE_FLAGS ${MKL_COMPILE_FLAGS})
    set(BLAS_LINK_FLAGS ${MKL_LINK_FLAGS})
    message(STATUS "Using Intel MKL")
else()
    if(WIN32)
        message(FATAL_ERROR "MKL not found. On Windows, MKL is required. Set MKLROOT environment variable.")
    endif()
    
    # Fall back to OpenBLAS (Linux only for now)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS openblas)
        pkg_check_modules(LAPACK lapack)
    endif()
    
    if(NOT OPENBLAS_FOUND)
        find_path(OPENBLAS_INCLUDE_DIRS cblas.h
            PATHS /usr/include /usr/include/openblas /usr/local/include)
        find_library(OPENBLAS_LIB openblas
            PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
        find_library(LAPACKE_LIB lapacke
            PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
        
        if(OPENBLAS_INCLUDE_DIRS AND OPENBLAS_LIB)
            set(OPENBLAS_FOUND TRUE)
            set(OPENBLAS_LIBRARIES ${OPENBLAS_LIB})
            if(LAPACKE_LIB)
                list(APPEND OPENBLAS_LIBRARIES ${LAPACKE_LIB})
            endif()
        endif()
    else()
        list(APPEND OPENBLAS_LIBRARIES lapacke)
    endif()
    
    if(OPENBLAS_FOUND)
        set(USE_OPENBLAS TRUE)
        set(BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIRS})
        set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
        message(STATUS "Using OpenBLAS")
    else()
        message(FATAL_ERROR "Neither MKL nor OpenBLAS found. Install one of them.")
    endif()
endif()

# Library sources
set(PCMCI_SOURCES
    src/dataframe.c
    src/parcorr.c
    src/skeleton.c
    src/mci.c
    src/significance.c
    src/pcmci.c
)

# Helper function to apply BLAS flags and definitions
function(apply_blas_flags target)
    if(BLAS_COMPILE_FLAGS)
        target_compile_options(${target} PRIVATE ${BLAS_COMPILE_FLAGS})
    endif()
    if(BLAS_LINK_FLAGS)
        target_link_options(${target} PRIVATE ${BLAS_LINK_FLAGS})
    endif()
    if(USE_MKL)
        target_compile_definitions(${target} PRIVATE PCMCI_USE_MKL)
    endif()
endfunction()

# Create shared library
add_library(pcmci SHARED ${PCMCI_SOURCES})
# Add .def file for Windows exports
if(WIN32)
    target_sources(pcmci PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pcmci.def)
endif()
target_include_directories(pcmci PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIRS}
)
target_link_libraries(pcmci PUBLIC 
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_C
)
# Define PCMCI_EXPORTS for DLL export (Windows)
target_compile_definitions(pcmci PRIVATE PCMCI_EXPORTS)
if(NOT WIN32)
    target_link_libraries(pcmci PUBLIC m)
    target_compile_definitions(pcmci PRIVATE _GNU_SOURCE)
endif()
apply_blas_flags(pcmci)

# Create static library
add_library(pcmci_static STATIC ${PCMCI_SOURCES})
target_include_directories(pcmci_static PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIRS}
)
target_link_libraries(pcmci_static PUBLIC 
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_C
)
if(NOT WIN32)
    target_link_libraries(pcmci_static PUBLIC m)
    target_compile_definitions(pcmci_static PRIVATE _GNU_SOURCE)
endif()
apply_blas_flags(pcmci_static)

# Example executable
add_executable(pcmci_example examples/example_basic.c)
if(WIN32)
    # On Windows, link statically to avoid DLL export complexity
    target_link_libraries(pcmci_example PRIVATE pcmci_static)
else()
    target_link_libraries(pcmci_example PRIVATE pcmci)
endif()
apply_blas_flags(pcmci_example)

# Synthetic data generator example
add_executable(pcmci_synthetic examples/example_synthetic.c)
if(WIN32)
    target_link_libraries(pcmci_synthetic PRIVATE pcmci_static)
else()
    target_link_libraries(pcmci_synthetic PRIVATE pcmci)
endif()
apply_blas_flags(pcmci_synthetic)

# Latency benchmark
add_executable(pcmci_benchmark examples/benchmark_latency.c)
if(WIN32)
    target_link_libraries(pcmci_benchmark PRIVATE pcmci_static)
else()
    target_link_libraries(pcmci_benchmark PRIVATE pcmci)
endif()
apply_blas_flags(pcmci_benchmark)

# Install targets
install(TARGETS pcmci pcmci_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER_ID} (${CMAKE_C_COMPILER})")
message(STATUS "BLAS include: ${BLAS_INCLUDE_DIRS}")
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
if(BLAS_COMPILE_FLAGS)
    message(STATUS "BLAS compile flags: ${BLAS_COMPILE_FLAGS}")
endif()
if(BLAS_LINK_FLAGS)
    message(STATUS "BLAS link flags: ${BLAS_LINK_FLAGS}")
endif()
message(STATUS "OpenMP found: ${OpenMP_C_FOUND}")

# Copy DLL to python folder for easy testing (Windows)
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python")
    add_custom_command(TARGET pcmci POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:pcmci>
        ${CMAKE_CURRENT_SOURCE_DIR}/python/
        COMMENT "Copying pcmci.dll to python folder"
    )
endif()