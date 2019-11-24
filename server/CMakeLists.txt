project(server_vo)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV 3.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread ")




include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/eigen3
    ${Eigen3_INCLUDE_DIRS}/eigen3
    ${Sophus_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    )


#add_library(utils
#    g2o_types.cpp
#    camera_model.cpp
#    )

add_executable(server_vo server_vo.cpp )
target_link_libraries(server_vo
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
#    /usr/local/lib/libSophus.so
#    utils g2o_core g2o_stuff g2o_types_sba
    )


#add_executable(vo vo.cpp g2o_types.cpp)
#target_link_libraries(vo  ${OpenCV_LIBS}  ${Sophus_LIBRARIES} /usr/local/lib/libSophus.so g2o_core utils  g2o_stuff g2o_types_sba )
