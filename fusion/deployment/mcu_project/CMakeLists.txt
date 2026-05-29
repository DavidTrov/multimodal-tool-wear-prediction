# CMakeLists.txt — FRDM-MCXN947 Tool Wear Prediction Pipeline
#
# This file is provided for reference and IDE integration. The canonical
# MCUXpresso IDE workflow uses the .mex / .cproject managed build system.
# Use this CMakeLists.txt with the MCUXpresso CMake export, arm-none-eabi-gcc,
# and the NXP MCUXpresso SDK CMake components.
#
# Prerequisites:
#   1. NXP MCUXpresso SDK for FRDM-MCXN947 (SDK_2_x_FRDM-MCXN947)
#      Download: https://mcuxpresso.nxp.com/
#   2. eIQ Middleware package (TFLite Micro + CMSIS-DSP included in SDK)
#   3. arm-none-eabi-gcc toolchain (≥ 12.x)
#
# Build (example — adjust MCUXPRESSO_SDK_ROOT to your install path):
#   cmake -DCMAKE_TOOLCHAIN_FILE=${MCUXPRESSO_SDK_ROOT}/tools/cmake/armgcc.cmake \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DMCUXPRESSO_SDK_ROOT=/path/to/SDK_2_x_FRDM-MCXN947 \
#         -B build -S .
#   cmake --build build

cmake_minimum_required(VERSION 3.20)
project(tool_wear_mcxn947 C CXX ASM)

# ── Toolchain / language standards ────────────────────────────────────────────
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 14)

# ── SDK root (set via -DMCUXPRESSO_SDK_ROOT=... or environment variable) ──────
if(NOT DEFINED MCUXPRESSO_SDK_ROOT)
    if(DEFINED ENV{MCUXPRESSO_SDK_ROOT})
        set(MCUXPRESSO_SDK_ROOT $ENV{MCUXPRESSO_SDK_ROOT})
    else()
        message(FATAL_ERROR
            "MCUXPRESSO_SDK_ROOT is not set.\n"
            "Pass -DMCUXPRESSO_SDK_ROOT=/path/to/SDK_2_x_FRDM-MCXN947")
    endif()
endif()

# ── Source files ───────────────────────────────────────────────────────────────
set(APP_SOURCES
    main.c
    sensor_pipeline.cpp
    # CWT preprocessor — copy from experiments/cwt_c/ into this directory
    cwt_mcu.c
)

# ── Include directories ────────────────────────────────────────────────────────
set(APP_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    # SDK board support
    ${MCUXPRESSO_SDK_ROOT}/boards/frdmmcxn947/board
    ${MCUXPRESSO_SDK_ROOT}/boards/frdmmcxn947/project_template
    # SDK device headers
    ${MCUXPRESSO_SDK_ROOT}/devices/MCXN947
    ${MCUXPRESSO_SDK_ROOT}/devices/MCXN947/drivers
    # CMSIS-DSP
    ${MCUXPRESSO_SDK_ROOT}/CMSIS/DSP/Include
    ${MCUXPRESSO_SDK_ROOT}/CMSIS/Core/Include
    # TFLite Micro (eIQ middleware)
    ${MCUXPRESSO_SDK_ROOT}/middleware/eiq/tensorflow-lite/tensorflow/lite/micro
    ${MCUXPRESSO_SDK_ROOT}/middleware/eiq/tensorflow-lite
    ${MCUXPRESSO_SDK_ROOT}/middleware/eiq/tensorflow-lite/third_party/flatbuffers/include
    ${MCUXPRESSO_SDK_ROOT}/middleware/eiq/tensorflow-lite/third_party/gemmlowp
    ${MCUXPRESSO_SDK_ROOT}/middleware/eiq/tensorflow-lite/third_party/ruy
)

# ── Compile definitions ────────────────────────────────────────────────────────
set(APP_DEFINITIONS
    CPU_MCXN947VDF_cm33_core0   # Device identifier for SDK headers
    ARM_MATH_CM33               # CMSIS-DSP Cortex-M33 path
    TF_LITE_STATIC_MEMORY       # TFLite Micro: no dynamic allocation
    NDEBUG                      # Disable assert() in release
    # Optional: uncomment to enable extended debug output from sensor_pipeline
    # SENSOR_PIPELINE_DEBUG=1
)

# ── Compiler flags ─────────────────────────────────────────────────────────────
set(MCU_FLAGS
    -mcpu=cortex-m33
    -mthumb
    -mfpu=fpv5-sp-d16
    -mfloat-abi=hard
)

set(COMMON_FLAGS
    ${MCU_FLAGS}
    -O2
    -ffast-math            # Enables reciprocal/sqrt approximations (safe for this use)
    -ffunction-sections    # Dead-strip unused functions
    -fdata-sections
    -Wall
    -Wno-unused-parameter
)

set(CMAKE_C_FLAGS_INIT   "${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS} -fno-exceptions -fno-rtti")

# ── Linker flags ───────────────────────────────────────────────────────────────
set(LINKER_SCRIPT ${MCUXPRESSO_SDK_ROOT}/boards/frdmmcxn947/project_template/MCXN947_flash.ld)
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "${MCU_FLAGS} \
     -T${LINKER_SCRIPT} \
     -Wl,--gc-sections \
     -Wl,-Map=tool_wear.map \
     --specs=nano.specs \
     --specs=nosys.specs")

# ── SDK library targets ────────────────────────────────────────────────────────
# MCUXpresso SDK exposes these via its own CMake infrastructure.
# Adjust target names to match the SDK version you have installed.
# If using MCUXpresso IDE's managed CMake export, these are already defined.
#
# Typical SDK libraries needed:
#   - sdk_board             (board.h, pin_mux.h, clock_config.h)
#   - sdk_drivers           (fsl_lpuart.h, fsl_debug_console.h, ...)
#   - sdk_cmsis_dsp         (arm_math.h, arm_cfft_f32, ...)
#   - eiq_tensorflow_lite   (TFLite Micro runtime)
#
# Example (SDK CMake component names vary by SDK version):
set(SDK_LIBS
    # Uncomment and adjust to match your SDK CMake targets:
    # nxp::drivers::lpuart
    # nxp::drivers::debug_console
    # nxp::cmsis_dsp
    # eiq::tensorflow_lite_micro
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(tool_wear.elf ${APP_SOURCES})

target_include_directories(tool_wear.elf PRIVATE ${APP_INCLUDES})
target_compile_definitions(tool_wear.elf PRIVATE ${APP_DEFINITIONS})

if(SDK_LIBS)
    target_link_libraries(tool_wear.elf PRIVATE ${SDK_LIBS})
endif()

# ── Post-build: generate binary and hex images ────────────────────────────────
find_program(OBJCOPY arm-none-eabi-objcopy REQUIRED)
find_program(SIZE    arm-none-eabi-size)

add_custom_command(TARGET tool_wear.elf POST_BUILD
    COMMAND ${OBJCOPY} -O binary tool_wear.elf tool_wear.bin
    COMMAND ${OBJCOPY} -O ihex   tool_wear.elf tool_wear.hex
    COMMAND ${SIZE} tool_wear.elf
    COMMENT "Generating binary and hex images"
)

# ── Flash target (requires pyOCD or J-Link) ───────────────────────────────────
# Flash via pyOCD:  cmake --build build --target flash
# Flash via J-Link: cmake --build build --target flash_jlink
#
# pyOCD target name for MCXN947: 'mcxn947'
# Install: pip install pyocd  then: pyocd pack install mcxn947
find_program(PYOCD pyocd)
if(PYOCD)
    add_custom_target(flash
        DEPENDS tool_wear.elf
        COMMAND ${PYOCD} flash -t mcxn947 tool_wear.hex
        COMMENT "Flashing via pyOCD"
    )
endif()

find_program(JLINK JLinkExe)
if(JLINK)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_jlink.jlink.in
        ${CMAKE_CURRENT_BINARY_DIR}/flash_jlink.jlink
        @ONLY
    )
    add_custom_target(flash_jlink
        DEPENDS tool_wear.elf
        COMMAND ${JLINK} -device MCXN947 -if SWD -speed 4000
                         -CommandFile ${CMAKE_CURRENT_BINARY_DIR}/flash_jlink.jlink
        COMMENT "Flashing via J-Link"
    )
endif()
