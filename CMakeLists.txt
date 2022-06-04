cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror" CACHE STRING "Default CXX options" FORCE)
set(CMAKE_CXX_STANDARD 20)

project(ml-in-sa CXX)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/)
set(EXPERIMENTS_DIR ${PROJECT_SOURCE_DIR}/experiments/)
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third-party/)
set(JULIET_TEST_SUITE_DIR ${THIRD_PARTY_DIR}/juliet-test-suite-c/)

add_subdirectory(src)
add_subdirectory(experiments)

