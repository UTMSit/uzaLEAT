# файл: Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -fopenmp -fPIC -Wall -Wextra -I.
LDFLAGS = -lpthread -ldl -fopenmp -rdynamic

# --- Vulkan detection ---------------------------------------------------------
# Приоритет: 1) pkg-config, 2) VULKAN_SDK env, 3) CPATH (NixOS)
HAS_VULKAN := 0
VULKAN_CFLAGS :=
VULKAN_LIBS :=

# 1) pkg-config
VULKAN_PC := $(shell pkg-config --cflags --libs vulkan 2>/dev/null)
ifneq ($(VULKAN_PC),)
    HAS_VULKAN := 1
    VULKAN_CFLAGS += -DUZALEAT_USE_VULKAN
    VULKAN_CFLAGS += $(shell pkg-config --cflags vulkan)
    VULKAN_LIBS += $(shell pkg-config --libs vulkan)
else
    # 2) VULKAN_SDK
    VULKAN_SDK_VAL ?= $(shell echo $$VULKAN_SDK)
    ifneq ($(VULKAN_SDK_VAL),)
        HAS_VULKAN := 1
        VULKAN_CFLAGS += -DUZALEAT_USE_VULKAN -I$(VULKAN_SDK_VAL)/include
        VULKAN_LIBS += -L$(VULKAN_SDK_VAL)/lib -lvulkan
    else
        # 3) CPATH / header check
        VULKAN_TEST := $(shell echo '\#include <vulkan/vulkan.h>' | $(CXX) -x c++ -E - 2>/dev/null && echo 1)
        ifeq ($(VULKAN_TEST),1)
            HAS_VULKAN := 1
            VULKAN_CFLAGS += -DUZALEAT_USE_VULKAN
            VULKAN_LIBS += -lvulkan
        endif
    endif
endif

CXXFLAGS += $(VULKAN_CFLAGS)
LDFLAGS += $(VULKAN_LIBS)

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
ifeq ($(HAS_VULKAN),1)
    SRCS += vulkan_backend.cpp
endif
OBJS = $(SRCS:.cpp=.o)

# Модель .so — собирается и без Vulkan (только CPU)
MODEL_SO = libslesa.so
MODEL_SRCS = slesa_flettohm.cpp
MODEL_OBJS = slesa_flettohm.o
ifeq ($(HAS_VULKAN),1)
    MODEL_OBJS += vulkan_backend.o
endif

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

$(MODEL_SO): $(MODEL_OBJS)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $^ $(LDFLAGS)

# vulkan_backend.o — только при доступном Vulkan SDK
vulkan_backend.o: vulkan_backend.cpp vulkan_backend.hpp
ifeq ($(HAS_VULKAN),1)
	$(CXX) $(CXXFLAGS) $(VULKAN_CFLAGS) -c $< -o $@
else
	@echo "Vulkan not available, skipping $@"
endif

clean:
	rm -f $(OBJS) vulkan_backend.o $(TARGET) $(MODEL_SO) slesa_flettohm.o
	rm -f $(SHADER_OUTS)
