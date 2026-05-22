# Makefile — CWT scalogram preprocessor (macOS / Linux, C99)
#
# Usage:
#   make          — build ./cwt_preprocess
#   make clean    — remove binary and object files
#
# Requires: clang (ships with macOS Xcode CLI tools) or gcc

CC      = clang
CFLAGS  = -O2 -std=c99 -Wall -Wextra -Wmissing-prototypes \
          -I. -Ikiss_fft
LDFLAGS = -lm

SRCS    = cwt_preprocess.c kiss_fft/kiss_fft.c main.c
OBJS    = $(SRCS:.c=.o)
TARGET  = cwt_preprocess

# MCU-optimised build (direct CWT, CMSIS-DSP shim for desktop)
MCU_SRCS    = cwt_mcu.c kiss_fft/kiss_fft.c main_mcu.c
MCU_OBJS    = $(MCU_SRCS:.c=.o)
MCU_TARGET  = cwt_mcu
MCU_CFLAGS  = $(CFLAGS) -DDESKTOP_SHIM

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(MCU_TARGET): cwt_mcu_d.o kiss_fft/kiss_fft_d.o main_mcu_d.o
	$(CC) $(MCU_CFLAGS) $^ -o $@ $(LDFLAGS)

# MCU objects use _d suffix to avoid colliding with desktop .o files
cwt_mcu_d.o: cwt_mcu.c
	$(CC) $(MCU_CFLAGS) -c $< -o $@

kiss_fft/kiss_fft_d.o: kiss_fft/kiss_fft.c
	$(CC) $(MCU_CFLAGS) -c $< -o $@

main_mcu_d.o: main_mcu.c
	$(CC) $(MCU_CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) $(MCU_TARGET) cwt_mcu_d.o kiss_fft/kiss_fft_d.o main_mcu_d.o
