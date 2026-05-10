# файл: Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -fopenmp -Wall -Wextra -I.
LDFLAGS = -lpthread -ldl -fopenmp

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

# Базовые шейдеры
SHADERS_BASE = matmul_small matmul_medium matmul_large \
               softmax wkv gelu sigmoid rms_norm ew_mul ew_add

# Оптимизированные под GCN (FP16 + subgroup) — только если поддерживается
SHADERS_GCN = matmul_fp16_large matmul_small_vec4 \
              softmax_subgroup wkv_fp16 rms_norm_subgroup gelu_fp16

ALL_SHADERS = $(SHADERS_BASE) $(SHADERS_GCN)
SHADER_SRCS = $(addprefix $(SHADER_DIR)/, $(addsuffix .comp, $(ALL_SHADERS)))
SHADER_OUTS = $(addprefix $(SHADER_DIR)/, $(addsuffix .spv, $(ALL_SHADERS)))

# Исходники основного проекта
SRCS = main.cpp uzaleat_core.cpp gutr_parser.cpp gutr_vm.cpp
ifneq ($(VULKAN_SDK),)
    SRCS += vulkan_backend.cpp
endif
OBJS = $(SRCS:.cpp=.o)

# Модель .so
MODEL_SO = libslesa.so
MODEL_SRCS = slesa_flettohm.cpp
MODEL_OBJS = $(MODEL_SRCS:.cpp=.o)

TARGET = uzaLEAT

# ============================================================================
# Цели
# ============================================================================
.PHONY: all clean shaders model

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Шейдеры
shaders: $(SHADER_OUTS)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp
	@mkdir -p $(SHADER_DIR)
	$(GLSLC) $(GLSLC_FLAGS) $< -o $@
	@echo "  SPIR-V  $@"

# Модель .so
model: shaders $(MODEL_SO)

$(MODEL_SO): $(MODEL_SRCS) vulkan_backend.o
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $^ \
		$(LDFLAGS) $(shell pkg-config --libs vulkan 2>/dev/null || echo "-lvulkan")

# Модель .so без GPU (fallback)
model-cpu: $(MODEL_SRCS)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $(MODEL_SO) $^ $(LDFLAGS) -lpthread

clean:
	rm -f $(OBJS) vulkan_backend.o $(TARGET) $(MODEL_SO)
	rm -f $(SHADER_OUTS)

info:
	@echo "VULKAN_SDK = $(VULKAN_SDK)"
	@echo "CXXFLAGS   = $(CXXFLAGS)"
	@echo "LDFLAGS    = $(LDFLAGS)"
	@echo "Shaders:   $(words $(SHADER_OUTS)) files"
