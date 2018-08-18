#CROSS_COMPILE = arm-linux-gnueabihf-
CC = $(CROSS_COMPILE)g++
TARGET = winograd_demo
CFLAGS = -std=c++11 -fPIC -lm -lpthread -lopenblas
#x86 version
$(TARGET): ./src/winograd_test.cpp
	$(CC) $^ -o $@ $(CFLAGS)

clean:
	-rm $(TARGET)
