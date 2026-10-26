set(TARGET_NAME LNLib)
project(${TARGET_NAME} LANGUAGES CXX)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_library(${TARGET_NAME} SHARED "")
target_compile_definitions(LNLib PRIVATE LNLIB_HOME)

target_include_directories(${TARGET_NAME} PRIVATE ${SOURCE_DIR}/include)

find_package(Eigen3 REQUIRED)
target_include_directories(${TARGET_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})

file(GLOB rootfiles *.cpp *.h)
source_group("" FILES ${rootfiles})
target_sources(${TARGET_NAME} PRIVATE ${rootfiles})
SUBDIRLIST(SUBDIRS ${SOURCE_DIR})
foreach(subdir ${SUBDIRS})
    file(GLOB subdirFiles ${subdir}/*.cpp ${subdir}/*.h)
    string(REPLACE "/" "\\" subdir ${subdir})
    source_group(${subdir} FILES ${subdirFiles})
    target_sources(${TARGET_NAME} PRIVATE ${subdirFiles})
endforeach()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/${TARGET_NAME}/bin/$<CONFIG>
        ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/${TARGET_NAME}/lib/$<CONFIG>)
install(DIRECTORY include DESTINATION ${CMAKE_BINARY_DIR}/${TARGET_NAME})
