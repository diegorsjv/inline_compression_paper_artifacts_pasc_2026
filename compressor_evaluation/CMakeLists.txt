cmake_minimum_required(VERSION 3.19)

# ---------------------------------------------------------------------------- #

IF(NOT ZFP_DIR)
    message("Please provide the path through ZFP (-DZFP_DIR)")
ENDIF()

# ---------------------------------------------------------------------------- #

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

enable_language(CXX C Fortran CUDA)

# Locate zfp paths ----------------------------------------------------------- #

set(ZFP_LIBRARY_DIR ${ZFP_DIR}/lib)
IF(EXISTS ${ZFP_LIBRARY_DIR})
    set(ZFP_LIBRARY_DIR ${ZFP_DIR}/lib)
    message("Found zfp lib at ${ZFP_LIBRARY_DIR}")
ELSEIF(EXISTS ${ZFP_DIR}/lib64)
    set(ZFP_LIBRARY_DIR ${ZFP_DIR}/lib64)
    message("Found zfp lib64 at ${ZFP_LIBRARY_DIR}")
ELSE()
    message("Check ZFP_DIR: Couldn't find zfp library at ${ZFP_LIBRARY_DIR}")
    message(FATAL_ERROR
            "Check ZFP_DIR: Couldn't find zfp library at ${ZFP_LIBRARY_DIR}64")
ENDIF()

set(ZFP_INCLUDE_DIR ${ZFP_DIR}/include)
IF(EXISTS ${ZFP_INCLUDE_DIR})
    message("Found zfp include at ${ZFP_INCLUDE_DIR}")
ELSE()
    message(FATAL_ERROR
            "Check ZFP_DIR: Couldn't find zfp include at ${ZFP_INCLUDE_DIR}")
ENDIF()

set(ZFP_MODULE_DIR ${ZFP_DIR}/modules)
IF(EXISTS ${ZFP_MODULE_DIR})
    message("Found zfp module at ${ZFP_MODULE_DIR}")
ELSE()
    message(FATAL_ERROR
            "Check ZFP_DIR: Couldn't find zfp module at ${ZFP_MODULE_DIR}")
ENDIF()


# ---------------------------------------------------------------------------- #
IF(cuSZp_DIR)
set(CUSZP_LIBRARY_DIR ${cuSZp_DIR}/lib)
IF(EXISTS ${CUSZP_LIBRARY_DIR})
    set(CUSZP_LIBRARY_DIR ${cuSZp_DIR}/lib)
    message("Found cuSZP lib at ${CUSZP_LIBRARY_DIR}")
ELSEIF(EXISTS ${cuSZp_DIR}/lib64)
    set(CUSZP_LIBRARY_DIR ${cuSZp_DIR}/lib64)
    message("Found cuSZp lib64 at ${CUSZP_LIBRARY_DIR}")
ELSE()
    message("Check CUSZP_DIR: Couldn't find cuSZP library at ${CUSZP_LIBRARY_DIR}")
    message(FATAL_ERROR
            "Check CUSZP_DIR: Couldn't find cuSZP library at ${CUSZP_LIBRARY_DIR}")
ENDIF()

set(CUSZP_INCLUDE_DIR ${cuSZp_DIR}/include)
IF(EXISTS ${CUSZP_INCLUDE_DIR})
    message("Found cuSZP include at ${CUSZP_INCLUDE_DIR}")
    set(CUSZP_INCLUDE_DIR ${cuSZp_DIR}/include)
ELSE()
    message(FATAL_ERROR
            "Check cuSZp_DIR: Couldn't find cuSZP include at ${CUSZP_INCLUDE_DIR}")
ENDIF()

ENDIF()
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
IF(MGARD_DIR)
    set(mgard_ROOT ${MGARD_DIR})
    list(APPEND CMAKE_PREFIX_PATH "${MGARD_DIR}")
ENDIF()
# ---------------------------------------------------------------------------- #

list(APPEND CMAKE_PREFIX_PATH "${ZFP_DIR}")

add_subdirectory(src)


unset(ZFP_DIR)
unset(CUSZP_DIR)
unset(MGARD_DIR)
