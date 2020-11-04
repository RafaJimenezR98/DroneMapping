cmake_minimum_required(VERSION 3.3.0)
project (openCV)
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV 4.3.0 REQUIRED)
LINK_LIBRARIES(${OpenCV_LIBS})


add_executable(mapeadoConDrones mapeadoConDrones.cpp)
