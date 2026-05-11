# файл: Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -fopenmp -fPIC -Wall -Wextra -I.
LDFLAGS = -lpthread -ldl -fopenmp -rdynamic

# Vulkan SDK
VULKAN_SDK ?= $(shell echo $$VULKAN_SDK)
ifneq ($(VULKAN_SDK),)
    VULKAN_INC = -I$(VULKAN_SDK)/include
    VULKAN_LIB = -L$(VULKAN_SDK)/lib
    CXXFLAGS += -DUZALEAT_USE_VULKAN $(VULKAN_INC)
    LDFLAGS += $(VULKAN_LIB) -lvulkan
endif

# Компилятор шейдеров
GLSLC = glslc
GLSLC_FLAGS = -O -fshader-stage=compute --target-env=vulkan1.2

SHADER_DIR = shaders
SHADERS_BASE = matmul_small matmul_medium matmul_large \
               softmax wkv gelu sigmoid rms_norm ew_mul ew_add
SHADER_SRCS = $(addprefix $(SHADER_DIR)/, $(addsuffix .comp, $(SHADERS_BASE)))
SHADER_OUTS = $(addprefix $(SHADER_DIR)/, $(addsuffix .spv, $(SHADERS_BASE)))

# Исходники основного проекта
SRCS = main.cpp uzaleat_core.cpp gutr_parser.cpp gutr_vm.cpp
ifneq ($(VULKAN_SDK),)
    SRCS += vulkan_backend.cpp
endif
OBJS = $(SRCS:.cpp=.o)

# Модель .so
MODEL_SO = libslesa.so
MODEL_SRCS = slesa_flettohm.cpp

TARGET = uzaLEAT

# ============================================================================
# Цели
# ============================================================================
.PHONY: all clean model

all: shaders $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Шейдеры
shaders: $(SHADER_OUTS)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp
	@mkdir -p $(SHADER_DIR)
	$(GLSLC) $(GLSLC_FLAGS) $< -o $@

# Модель .so
model: shaders $(MODEL_SO)

$(MODEL_SO): $(MODEL_SRCS) vulkan_backend.o
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $^ \
		$(LDFLAGS) $(shell pkg-config --libs vulkan 2>/dev/null || echo "-lvulkan")

# vulkan_backend.o — всегда с PIC и UZALEAT_USE_VULKAN
vulkan_backend.o: vulkan_backend.cpp vulkan_backend.hpp
	$(CXX) -std=c++17 -O3 -fopenmp -fPIC -DUZALEAT_USE_VULKAN -I. -c vulkan_backend.cpp -o vulkan_backend.o

clean:
	rm -f $(OBJS) vulkan_backend.o $(TARGET) $(MODEL_SO)
	rm -f $(SHADER_OUTS)
