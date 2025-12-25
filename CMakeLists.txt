cmake_minimum_required(VERSION 3.16)
project(pcmci C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Detect compiler
if(CMAKE_C_COMPILER_ID MATCHES "Intel" OR CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(USING_INTEL_COMPILER TRUE)
    message(STATUS "Detected Intel compiler: ${CMAKE_C_COMPILER_ID}")
else()
    set(USING_INTEL_COMPILER FALSE)
endif()

# Compiler flags
if(USING_INTEL_COMPILER)
    # Intel ICX flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -debug full -DDEBUG")
    # Aggressive optimization for Intel: AVX-512, interprocedural optimization
    set(CMAKE_C_FLAGS_RELEASE "-O3 -xHost -ipo -DNDEBUG")
    # Optional: Add -qopt-report=5 for optimization reports
    
    # For different Intel CPU targets, you can use:
    # -xCORE-AVX512 for Skylake-X/Cascade Lake
    # -xICELAKE-SERVER for Ice Lake
    # -xSAPPHIREPARIDS for Sapphire Rapids  
    # -xHost auto-detects the build machine
else()
    # GCC/Clang flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find BLAS/LAPACK - try MKL first, fall back to OpenBLAS
find_package(PkgConfig QUIET)

set(USE_MKL FALSE)
set(USE_OPENBLAS FALSE)

# Try MKL first (preferred with Intel compiler)
if(PkgConfig_FOUND AND NOT USING_INTEL_COMPILER)
    pkg_check_modules(MKL QUIET mkl-dynamic-lp64-iomp)
endif()

if(NOT MKL_FOUND)
    set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "MKL root directory")
    if(NOT MKL_ROOT)
        set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
    endif()
    
    if(EXISTS "${MKL_ROOT}/include/mkl.h")
        set(MKL_INCLUDE_DIRS "${MKL_ROOT}/include")
        set(MKL_LIBRARY_DIRS "${MKL_ROOT}/lib/intel64")
        
        if(USING_INTEL_COMPILER)
            # Intel compiler: use -qmkl flag for automatic linking
            # This is the cleanest way with ICX
            set(MKL_LIBRARIES "-qmkl=parallel")
            set(MKL_COMPILE_FLAGS "-qmkl=parallel")
        else()
            # GCC/Clang: explicit linking
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
        endif()
        set(MKL_FOUND TRUE)
    endif()
endif()

if(MKL_FOUND)
    set(USE_MKL TRUE)
    set(BLAS_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLAS_COMPILE_FLAGS ${MKL_COMPILE_FLAGS})
    message(STATUS "Using Intel MKL")
else()
    # Fall back to OpenBLAS
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS openblas)
        pkg_check_modules(LAPACK lapack)
    endif()
    
    if(NOT OPENBLAS_FOUND)
        # Manual OpenBLAS detection
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
        # pkg-config found it, but we still need lapacke
        list(APPEND OPENBLAS_LIBRARIES lapacke)
    endif()
    
    if(OPENBLAS_FOUND)
        set(USE_OPENBLAS TRUE)
        set(BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIRS})
        set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
        set(BLAS_COMPILE_FLAGS "")
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

# Create shared library
add_library(pcmci SHARED ${PCMCI_SOURCES})
target_include_directories(pcmci PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIRS}
)
target_link_libraries(pcmci PUBLIC 
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_C
    m
)
target_compile_definitions(pcmci PRIVATE _GNU_SOURCE)
if(BLAS_COMPILE_FLAGS)
    target_compile_options(pcmci PRIVATE ${BLAS_COMPILE_FLAGS})
    target_link_options(pcmci PRIVATE ${BLAS_COMPILE_FLAGS})
endif()
if(USE_MKL)
    target_compile_definitions(pcmci PRIVATE PCMCI_USE_MKL)
endif()

# Create static library
add_library(pcmci_static STATIC ${PCMCI_SOURCES})
target_include_directories(pcmci_static PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIRS}
)
target_link_libraries(pcmci_static PUBLIC 
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_C
    m
)
target_compile_definitions(pcmci_static PRIVATE _GNU_SOURCE)
if(BLAS_COMPILE_FLAGS)
    target_compile_options(pcmci_static PRIVATE ${BLAS_COMPILE_FLAGS})
    target_link_options(pcmci_static PRIVATE ${BLAS_COMPILE_FLAGS})
endif()
if(USE_MKL)
    target_compile_definitions(pcmci_static PRIVATE PCMCI_USE_MKL)
endif()

# Example executable
add_executable(pcmci_example examples/example_basic.c)
target_link_libraries(pcmci_example PRIVATE pcmci)
if(BLAS_COMPILE_FLAGS)
    target_compile_options(pcmci_example PRIVATE ${BLAS_COMPILE_FLAGS})
    target_link_options(pcmci_example PRIVATE ${BLAS_COMPILE_FLAGS})
endif()

# Synthetic data generator example
add_executable(pcmci_synthetic examples/example_synthetic.c)
target_link_libraries(pcmci_synthetic PRIVATE pcmci)
if(BLAS_COMPILE_FLAGS)
    target_compile_options(pcmci_synthetic PRIVATE ${BLAS_COMPILE_FLAGS})
    target_link_options(pcmci_synthetic PRIVATE ${BLAS_COMPILE_FLAGS})
endif()

# Install targets
install(TARGETS pcmci pcmci_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER_ID} (${CMAKE_C_COMPILER})")
message(STATUS "BLAS include: ${BLAS_INCLUDE_DIRS}")
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
if(BLAS_COMPILE_FLAGS)
    message(STATUS "BLAS compile flags: ${BLAS_COMPILE_FLAGS}")
endif()
message(STATUS "OpenMP found: ${OpenMP_C_FOUND}")