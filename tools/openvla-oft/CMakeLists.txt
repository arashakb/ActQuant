# OpenVLA-OFT implementation for llama.cpp

find_package(Threads REQUIRED)

OPTION(BUILD_PYTHON "Build Python bindings" OFF)
message(STATUS "Building Python bindings: ${BUILD_PYTHON}")

set(TARGET openvla_oft)

set(SRC_FILES ctx_manager.cpp build_graph.cpp llm.cpp model_loader.cpp model_defs.cpp utils.cpp openvla.cpp)
if(BUILD_PYTHON)
    list(APPEND SRC_FILES interface.cpp)
    add_library(${TARGET} SHARED ${SRC_FILES})
    set_target_properties(${TARGET} PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python: ${Python_EXECUTABLE} (found version \"${Python_VERSION}\")")
    message(STATUS "Python Include Dir: ${Python_INCLUDE_DIRS} Python Libraries: ${Python_LIBRARIES}")

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11, sys; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 CONFIG REQUIRED)
    target_include_directories(${TARGET} PRIVATE ${Python_INCLUDE_DIRS})
    target_link_directories(${TARGET} PRIVATE ${Python_LIBRARIES})
    target_link_libraries(${TARGET} PRIVATE pybind11::module)
    target_compile_definitions(${TARGET} PRIVATE BUILD_PYTHON_BINDINGS)
else()
    list(APPEND SRC_FILES main.cpp)
    add_executable(${TARGET} ${SRC_FILES})
endif()

target_link_libraries     (${TARGET} PUBLIC ggml llama)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_include_directories(${TARGET} PUBLIC  .)
target_include_directories(${TARGET} PRIVATE ../..)
target_include_directories(${TARGET} PRIVATE ../../vendor)
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC  LLAMA_SHARED)
endif()

install(TARGETS ${TARGET} LIBRARY)

if (NOT MSVC)
    target_compile_options(${TARGET} PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
endif()

set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME ${TARGET})
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common Threads::Threads tokenizers_cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
