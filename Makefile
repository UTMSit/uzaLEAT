# файл: Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -I.
LDFLAGS = -lpthread -ldl -fopenmp

VULKAN_SDK ?= $(shell echo $$VULKAN_SDK)
ifneq ($(VULKAN_SDK),)
    VULKAN_INC = -I$(VULKAN_SDK)/include
    VULKAN_LIB = -L$(VULKAN_SDK)/lib
    VULKAN_LIBS = -lvulkan -lshaderc_shared
    CXXFLAGS += -DUZALEAT_USE_VULKAN $(VULKAN_INC)
    LDFLAGS += $(VULKAN_LIB) $(VULKAN_LIBS)
endif

SRCS = main.cpp uzaleat_core.cpp gutr_parser.cpp gutr_vm.cpp
ifneq ($(VULKAN_SDK),)
    SRCS += vulkan_backend.cpp
endif
OBJS = $(SRCS:.cpp=.o)
TARGET = uzaLEAT

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
