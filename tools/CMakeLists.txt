# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # ActQuant: only the targets needed to reproduce the paper are built.
    # Upstream llama.cpp targets that aren't on the paper's critical path
    # are intentionally left out to keep the source tree minimal.
    add_subdirectory(../vendor/tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)
    add_subdirectory(openvla-oft)     # paper backbone #1
    add_subdirectory(pi0.5)           # paper backbone #2
    add_subdirectory(quantize)        # llama-quantize (HSIC + AMF imatrix consumer)
    add_subdirectory(quantize-vision) # vision-tower quantization
endif()
