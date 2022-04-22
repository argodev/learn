cmake_minimum_required (VERSION 3.6)
project (curie_test)

# Version info
set(curie_test_VERSION_MAJOR 0)
set(curie_test_VERSION_MINOR 0)
set(curie_test_VERSION_PATCH 0)
set(curie_test_VERSION ${curie_test_VERSION_MAJOR}.${curie_test_VERSION_MINOR}.${curie_test_VERSION_PATCH})

# Compiler Options
set (CMAKE_CXX_STANDARD 11)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)


# Boost Options
set(Boost_USE_MULTITHREADED    ON)
set(Boost_USE_STATIC_LIBS      ON)
set(Boost_USE_STATIC_RUNTIME   ON)

# first we can indicate the documentation build as an option and set it to ON by default
option(BUILD_DOC "Build documentation" ON)

# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

# let's add support for Eigen
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)


# Dependencies
find_package(Boost 1.58 COMPONENTS program_options log REQUIRED)
find_package(Threads REQUIRED)

include_directories(
	include
	${Boost_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH})

add_executable(MatVec
	src/MatVec.cpp)

add_executable(MatVecMPI
	src/MatVecMPI.cpp)

target_link_libraries(MatVec
	${CMAKE_THREAD_LIBS_INIT}
	${Boost_SYSTEM_LIBRARY}
	${Boost_PROGRAM_OPTIONS_LIBRARY}
	${Boost_PROGRAM_Log_LIBRARY}
	${Boost_LIBRARIES})

target_link_libraries(MatVecMPI
	${CMAKE_THREAD_LIBS_INIT}
	${Boost_SYSTEM_LIBRARY}
	${Boost_PROGRAM_OPTIONS_LIBRARY}
	${Boost_PROGRAM_Log_LIBRARY}
	${Boost_LIBRARIES}
    ${MPI_LIBRARIES})
