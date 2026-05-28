find_package(Threads REQUIRED)

# ============================================================================
# Shared core library (avoids recompiling common code for each target)
# ============================================================================
set(CORE_SRC_FILES
    ctx_manager.cpp
    build_graph.cpp
    model_loader.cpp
    model_defs.cpp
    utils.cpp
    pi05.cpp
    pi05_memory.cpp
    pi05_context.cpp
)
# Note: llm.cpp removed - using text_embed.hpp (header-only) for text embedding

add_library(pi05_core OBJECT ${CORE_SRC_FILES})
target_link_libraries(pi05_core PUBLIC ggml llama)
target_include_directories(pi05_core PUBLIC .)
target_include_directories(pi05_core PRIVATE ../..)
target_include_directories(pi05_core PRIVATE ../../src)
target_include_directories(pi05_core PRIVATE ../../vendor)
target_include_directories(pi05_core PRIVATE ../../ggml/src)
target_compile_features(pi05_core PRIVATE cxx_std_17)
set_target_properties(pi05_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT MSVC)
    target_compile_options(pi05_core PRIVATE -Wno-cast-qual)
endif()

# Tokenizer options
option(USE_SENTENCEPIECE "Use SentencePiece tokenizer (same as OpenPI)" ON)
option(USE_TOKENIZERS_CPP "Use tokenizers_cpp for HuggingFace tokenizer support (requires Rust)" OFF)

if(USE_SENTENCEPIECE)
    target_include_directories(pi05_core PRIVATE ${CMAKE_SOURCE_DIR}/vendor/tokenizers-cpp/sentencepiece/src)
    target_link_libraries(pi05_core PRIVATE common Threads::Threads sentencepiece-static)
    target_compile_definitions(pi05_core PRIVATE USE_SENTENCEPIECE)
    message(STATUS "Pi0.5: Using SentencePiece tokenizer (same as OpenPI)")
elseif(USE_TOKENIZERS_CPP)
    target_link_libraries(pi05_core PRIVATE common Threads::Threads tokenizers_cpp)
    target_compile_definitions(pi05_core PRIVATE USE_TOKENIZERS_CPP)
    message(STATUS "Pi0.5: Using tokenizers_cpp (HuggingFace tokenizer)")
else()
    target_link_libraries(pi05_core PRIVATE common Threads::Threads)
    message(STATUS "Pi0.5: Using llama built-in vocab (may have tokenization issues)")
endif()

if (TARGET BUILD_INFO)
    add_dependencies(pi05_core BUILD_INFO)
endif()

# ============================================================================
# Main executable (CLI tool)
# ============================================================================
add_executable(pi05 main.cpp $<TARGET_OBJECTS:pi05_core>)
target_link_libraries(pi05 PUBLIC ggml llama)
target_link_libraries(pi05 PRIVATE Threads::Threads)
target_include_directories(pi05 PUBLIC .)
target_include_directories(pi05 PRIVATE ../..)
target_include_directories(pi05 PRIVATE ../../ggml/src)
target_compile_features(pi05 PRIVATE cxx_std_17)

if(USE_SENTENCEPIECE)
    target_include_directories(pi05 PRIVATE ${CMAKE_SOURCE_DIR}/vendor/tokenizers-cpp/sentencepiece/src)
    target_link_libraries(pi05 PRIVATE common sentencepiece-static)
    target_compile_definitions(pi05 PRIVATE USE_SENTENCEPIECE)
elseif(USE_TOKENIZERS_CPP)
    target_link_libraries(pi05 PRIVATE common tokenizers_cpp)
    target_compile_definitions(pi05 PRIVATE USE_TOKENIZERS_CPP)
else()
    target_link_libraries(pi05 PRIVATE common)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS pi05 RUNTIME)
endif()

# ============================================================================
# Verification tool (disabled - source file not present)
# ============================================================================
# add_executable(verify_ggml verify_ggml.cpp $<TARGET_OBJECTS:pi05_core>)
# target_link_libraries(verify_ggml PUBLIC ggml llama)
# target_link_libraries(verify_ggml PRIVATE Threads::Threads)
# target_include_directories(verify_ggml PUBLIC .)
# target_include_directories(verify_ggml PRIVATE ../..)
# target_compile_features(verify_ggml PRIVATE cxx_std_17)
#
# if(USE_SENTENCEPIECE)
#     target_include_directories(verify_ggml PRIVATE ${CMAKE_SOURCE_DIR}/vendor/tokenizers-cpp/sentencepiece/src)
#     target_link_libraries(verify_ggml PRIVATE common sentencepiece-static)
#     target_compile_definitions(verify_ggml PRIVATE USE_SENTENCEPIECE)
# elseif(USE_TOKENIZERS_CPP)
#     target_link_libraries(verify_ggml PRIVATE common tokenizers_cpp)
#     target_compile_definitions(verify_ggml PRIVATE USE_TOKENIZERS_CPP)
# else()
#     target_link_libraries(verify_ggml PRIVATE common)
# endif()

# ============================================================================
# Python Module (for serve_policy.py)
# ============================================================================
option(BUILD_PI05_PYTHON "Build Pi0.5 Python module" ON)
if(BUILD_PI05_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Pi0.5 Python: Found Python ${Python_VERSION}")

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
    )

    if(NOT PYBIND11_RESULT EQUAL 0)
        message(WARNING "Pi0.5 Python: pybind11 not found via Python, trying system pybind11")
        find_package(pybind11 CONFIG)
    else()
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    if(pybind11_FOUND)
        add_library(pi05_py SHARED interface.cpp $<TARGET_OBJECTS:pi05_core>)
        set_target_properties(pi05_py PROPERTIES
            PREFIX ""
            OUTPUT_NAME "pi05"
            SUFFIX ".so"
            POSITION_INDEPENDENT_CODE ON
        )

        target_link_libraries(pi05_py PUBLIC ggml llama)
        target_link_libraries(pi05_py PRIVATE Threads::Threads pybind11::module)
        target_include_directories(pi05_py PUBLIC .)
        target_include_directories(pi05_py PRIVATE ../..)
        target_include_directories(pi05_py PRIVATE ../../src)
        target_include_directories(pi05_py PRIVATE ../../ggml/src)
        target_include_directories(pi05_py PRIVATE ${Python_INCLUDE_DIRS})
        target_compile_features(pi05_py PRIVATE cxx_std_17)
        target_compile_definitions(pi05_py PRIVATE BUILD_PYTHON_BINDINGS)

        if (NOT MSVC)
            target_compile_options(pi05_py PRIVATE -Wno-cast-qual)
        endif()

        if(USE_SENTENCEPIECE)
            target_include_directories(pi05_py PRIVATE ${CMAKE_SOURCE_DIR}/vendor/tokenizers-cpp/sentencepiece/src)
            target_link_libraries(pi05_py PRIVATE common sentencepiece-static)
            target_compile_definitions(pi05_py PRIVATE USE_SENTENCEPIECE)
        elseif(USE_TOKENIZERS_CPP)
            target_link_libraries(pi05_py PRIVATE common tokenizers_cpp)
            target_compile_definitions(pi05_py PRIVATE USE_TOKENIZERS_CPP)
        else()
            target_link_libraries(pi05_py PRIVATE common)
        endif()

        message(STATUS "Pi0.5: Python module target added (pi05.so)")
    else()
        message(WARNING "Pi0.5: pybind11 not found, Python module will not be built")
    endif()
endif()
