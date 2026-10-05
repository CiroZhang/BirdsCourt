cmake_minimum_required(VERSION 3.5)
project(detect)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Some OpenCV distributions (e.g. certain HPC/EasyBuild installs) ship a
# broken opencv_hdf module (missing HDF5 symbols at link time) that this
# project never actually uses -- drop it from the link line if present
# rather than making every environment debug the same linker error.
list(FILTER OpenCV_LIBS EXCLUDE REGEX "hdf")

file(GLOB SRC_FILES *.cpp)
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
