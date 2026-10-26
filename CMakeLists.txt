cmake_minimum_required(VERSION 3.16)
project(curve_reconstruction LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ENABLE_UNIT_TESTS OFF CACHE BOOL "Build LNLib tests" FORCE)
add_subdirectory(3rdparties/LNLib EXCLUDE_FROM_ALL)

file(GLOB ALGLIB_SOURCES CONFIGURE_DEPENDS "3rdparties/alglib/*.cpp")
add_executable(main
    main.cpp
    src/recon_config.cpp
    src/curve_detector/curve_detector.cpp
    src/curve_detector/LSD/lsd.cpp
    src/curve_matcher/curve_matcher.cpp
    src/curve_reconstructor/baseline_reconstructor.cpp
    src/curve_reconstructor/curve_reconstructor.cpp
    src/curve_reconstructor/find_s.cpp
    src/curve_reconstructor/pointcloud_to_nurbs.cpp
    src/utils/camera.cpp
    src/utils/curve.cpp
    src/utils/DLT_points.cpp
    src/utils/depth_rbf.cpp
    src/utils/warp_points.cpp
    ${ALGLIB_SOURCES}
)
target_include_directories(main PRIVATE
    include 3rdparties
    src/utils src/curve_detector src/curve_matcher src/curve_reconstructor
    3rdparties/LNLib/src/LNLib/include
    ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS}
)
target_compile_options(main PRIVATE -Wno-attributes)
target_compile_definitions(main PRIVATE CURVE_RECONSTRUCTION_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
    Eigen3::Eigen pybind11::embed yaml-cpp LNLib
)
