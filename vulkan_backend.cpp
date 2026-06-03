// ============================================================================
// vulkan_backend.cpp — ПОЛНАЯ РЕАЛИЗАЦИЯ
// Ни одной заглушки. Ни одного "// ...". Ни одного "остальной код".
// Каждый метод реализован полностью.
// ============================================================================

#include "vulkan_backend.hpp"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>

uzagpt::VulkanBackend* g_vk_global = nullptr;

namespace uzagpt {

// ============================================================================
// ЗАГРУЗКА SPIR-V ИЗ ФАЙЛА
// ============================================================================
static std::vector<uint32_t> load_spirv_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "VulkanBackend: Cannot open SPIR-V file: " << path << std::endl;
        return {};
    }
    size_t fileSize = file.tellg();
    if (fileSize % 4 != 0) {
        std::cerr << "VulkanBackend: SPIR-V file size not multiple of 4: " << path << std::endl;
        return {};
    }
    file.seekg(0);
    std::vector<uint32_t> spirv(fileSize / 4);
    file.read(reinterpret_cast<char*>(spirv.data()), fileSize);
    file.close();
    return spirv;
}

// ============================================================================
// Конструктор / Деструктор
// ============================================================================
VulkanBackend::VulkanBackend() {
    // Убрать memset — он ломает std::list, std::mutex, std::vector
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    transfer_queue_ = VK_NULL_HANDLE;
    compute_family_ = 0;
    transfer_family_ = 0;
    cmd_pool_ = VK_NULL_HANDLE;
    current_cmd_ = VK_NULL_HANDLE;
    batch_fence_ = VK_NULL_HANDLE;
    batch_open_ = false;
    transfer_cmd_pool_ = VK_NULL_HANDLE;
    transfer_cmd_ = VK_NULL_HANDLE;
    transfer_fence_ = VK_NULL_HANDLE;
    desc_pool_ = VK_NULL_HANDLE;
    desc_layout_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    matmul_small_pipeline_ = VK_NULL_HANDLE;
    matmul_medium_pipeline_ = VK_NULL_HANDLE;
    matmul_large_pipeline_ = VK_NULL_HANDLE;
    softmax_pipeline_ = VK_NULL_HANDLE;
    wkv_pipeline_ = VK_NULL_HANDLE;
    gelu_pipeline_ = VK_NULL_HANDLE;
    sigmoid_pipeline_ = VK_NULL_HANDLE;
    rms_norm_pipeline_ = VK_NULL_HANDLE;
    ew_mul_pipeline_ = VK_NULL_HANDLE;
    ew_add_pipeline_ = VK_NULL_HANDLE;
    pool_total_ = 0;
    pool_used_ = 0;
    staging_used_ = 0;
    staging_offset_ = 0;
    max_workgroup_size_ = 256;
    max_workgroup_invocations_ = 1024;
    max_shared_memory_ = 32768;
    has_fp16_ = false;
    has_async_compute_ = false;
}

VulkanBackend::~VulkanBackend() {
    shutdown();
}

// ============================================================================
// Инициализация
// ============================================================================
bool VulkanBackend::init(const GPUConfig& cfg) {
    config_ = cfg;

    create_instance();
    if (instance_ == VK_NULL_HANDLE) return false;

    pick_physical_device();
    if (physical_device_ == VK_NULL_HANDLE) return false;

    create_logical_device();
    if (device_ == VK_NULL_HANDLE) return false;

    create_queues();
    create_command_pools();

    if (cmd_pool_ == VK_NULL_HANDLE) return false;
    if (config_.enable_async_transfer && transfer_family_ != compute_family_) {
        if (transfer_cmd_pool_ == VK_NULL_HANDLE) return false;
    }

    create_descriptor_layout();
    if (desc_layout_ == VK_NULL_HANDLE) return false;

    create_pipeline_layout();
    if (pipeline_layout_ == VK_NULL_HANDLE) return false;

    if (!load_shaders_from_files()) return false;
    if (!create_all_pipelines()) return false;

    allocate_buffer_pool();
    allocate_staging_buffer();

    print_device_info();
    return true;
}

void VulkanBackend::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device_);

    // 0. Очистить пул буферов
    for (auto& entry : buffer_pool_) {
        if (entry.buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, entry.buffer.buffer, nullptr);
            entry.buffer.buffer = VK_NULL_HANDLE;
        }
        if (entry.buffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, entry.buffer.memory, nullptr);
            entry.buffer.memory = VK_NULL_HANDLE;
        }
    }
    buffer_pool_.clear();

    // 1. Удалить staging
    if (staging_buffer_.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, staging_buffer_.buffer, nullptr);
        staging_buffer_.buffer = VK_NULL_HANDLE;
    }
    if (staging_buffer_.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, staging_buffer_.memory, nullptr);
        staging_buffer_.memory = VK_NULL_HANDLE;
    }

    // 2. Пайплайны
    auto destroyPipeline = [this](VkPipeline& p) {
        if (p != VK_NULL_HANDLE) { vkDestroyPipeline(device_, p, nullptr); p = VK_NULL_HANDLE; }
    };
    destroyPipeline(matmul_small_pipeline_);
    destroyPipeline(matmul_medium_pipeline_);
    destroyPipeline(matmul_large_pipeline_);
    destroyPipeline(softmax_pipeline_);
    destroyPipeline(wkv_pipeline_);
    destroyPipeline(gelu_pipeline_);
    destroyPipeline(sigmoid_pipeline_);
    destroyPipeline(rms_norm_pipeline_);
    destroyPipeline(ew_mul_pipeline_);
    destroyPipeline(ew_add_pipeline_);

    if (pipeline_layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr); pipeline_layout_ = VK_NULL_HANDLE; }
    if (desc_pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, desc_pool_, nullptr); desc_pool_ = VK_NULL_HANDLE; }
    if (desc_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr); desc_layout_ = VK_NULL_HANDLE; }
    if (batch_fence_ != VK_NULL_HANDLE) { vkDestroyFence(device_, batch_fence_, nullptr); batch_fence_ = VK_NULL_HANDLE; }
    if (transfer_fence_ != VK_NULL_HANDLE) { vkDestroyFence(device_, transfer_fence_, nullptr); transfer_fence_ = VK_NULL_HANDLE; }
    if (cmd_pool_ != VK_NULL_HANDLE) { vkDestroyCommandPool(device_, cmd_pool_, nullptr); cmd_pool_ = VK_NULL_HANDLE; }

    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;

    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;

    physical_device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    transfer_queue_ = VK_NULL_HANDLE;
    current_cmd_ = VK_NULL_HANDLE;
    transfer_cmd_ = VK_NULL_HANDLE;
    batch_open_ = false;
}

// ============================================================================
// Создание Instance
// ============================================================================
void VulkanBackend::create_instance() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "uzaLEAT GPU Backend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "uzaLEAT";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    const char* extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    uint32_t extCount = 1;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = extCount;
    createInfo.ppEnabledExtensionNames = extensions;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create instance, error " << result << std::endl;
        instance_ = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Выбор физического устройства
// ============================================================================
void VulkanBackend::pick_physical_device() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "VulkanBackend: No physical devices found" << std::endl;
        return;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    physical_device_ = devices[0];
    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = devices[i];
            break;
        }
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);

    has_fp16_ = false;

    max_workgroup_size_ = props.limits.maxComputeWorkGroupSize[0];
    max_workgroup_invocations_ = props.limits.maxComputeWorkGroupInvocations;
    max_shared_memory_ = props.limits.maxComputeSharedMemorySize;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount, queueFamilies.data());

    compute_family_ = UINT32_MAX;
    transfer_family_ = UINT32_MAX;

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (compute_family_ == UINT32_MAX) compute_family_ = i;
        }
        if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
            if (transfer_family_ == UINT32_MAX && i != compute_family_) {
                transfer_family_ = i;
            }
        }
    }

    if (transfer_family_ == UINT32_MAX) {
        transfer_family_ = compute_family_;
        has_async_compute_ = false;
    } else {
        has_async_compute_ = true;
    }

    std::cout << "VulkanBackend: Selected " << props.deviceName << std::endl;
}

// ============================================================================
// Создание логического устройства
// ============================================================================
void VulkanBackend::create_logical_device() {
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo computeQueueCreateInfo = {};
    computeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQueueCreateInfo.queueFamilyIndex = compute_family_;
    computeQueueCreateInfo.queueCount = 1;
    computeQueueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.push_back(computeQueueCreateInfo);
    // НЕ создаём transfer очередь — используем compute для всего

    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;

    VkResult result = vkCreateDevice(physical_device_, &createInfo, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create logical device, error " << result << std::endl;
        device_ = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Получение очередей
// ============================================================================
void VulkanBackend::create_queues() {
    vkGetDeviceQueue(device_, compute_family_, 0, &compute_queue_);
    transfer_queue_ = compute_queue_;  // одна очередь для всего
}

// ============================================================================
// Командные пулы и буферы
// ============================================================================
void VulkanBackend::create_command_pools() {
    VkCommandPoolCreateInfo computePoolInfo = {};
    computePoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    computePoolInfo.queueFamilyIndex = compute_family_;
    computePoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult result = vkCreateCommandPool(device_, &computePoolInfo, nullptr, &cmd_pool_);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create compute command pool" << std::endl;
        return;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmd_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(device_, &allocInfo, &current_cmd_);

    // ВСЕГДА используем compute pool — без отдельных transfer очередей
    transfer_cmd_pool_ = cmd_pool_;
    transfer_cmd_ = current_cmd_;
    transfer_family_ = compute_family_;  // ВАЖНО: одна семья для всего

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vkCreateFence(device_, &fenceInfo, nullptr, &batch_fence_);
    vkCreateFence(device_, &fenceInfo, nullptr, &transfer_fence_);
}

// ============================================================================
// Дескрипторы
// ============================================================================
void VulkanBackend::create_descriptor_layout() {
    VkDescriptorSetLayoutBinding bindings[6];
    for (uint32_t i = 0; i < 6; i++) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 6;
    layoutInfo.pBindings = bindings;

    vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &desc_layout_);
}

void VulkanBackend::create_pipeline_layout() {
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 5 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &desc_layout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipeline_layout_);

    uint32_t maxSets = config_.max_buffers > 0 ? config_.max_buffers : 256;

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = maxSets * 6;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkResult result = vkCreateDescriptorPool(device_, &poolInfo, nullptr, &desc_pool_);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create descriptor pool with " << maxSets << " sets" << std::endl;
        return;
    }

    desc_slots_.resize(maxSets);
    std::vector<VkDescriptorSetLayout> layouts(maxSets, desc_layout_);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = desc_pool_;
    allocInfo.descriptorSetCount = maxSets;
    allocInfo.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> sets(maxSets);
    vkAllocateDescriptorSets(device_, &allocInfo, sets.data());

    for (uint32_t i = 0; i < maxSets; i++) {
        desc_slots_[i].set = sets[i];
        desc_slots_[i].buffer_infos.assign(6, {});
        desc_slots_[i].in_use = false;
    }
}

// ============================================================================
// Загрузка шейдеров из файлов
// ============================================================================
bool VulkanBackend::load_shaders_from_files() {
    auto try_load = [&](std::vector<uint32_t>& spv, const char* name) -> bool {
        spv = load_spirv_file(name);
        return !spv.empty();
    };

    bool all_found = true;
    all_found &= try_load(matmul_small_spv_,  "shaders/matmul_small.spv");
    all_found &= try_load(matmul_medium_spv_, "shaders/matmul_medium.spv");
    all_found &= try_load(matmul_large_spv_,  "shaders/matmul_large.spv");
    all_found &= try_load(softmax_spv_,       "shaders/softmax.spv");
    all_found &= try_load(wkv_spv_,           "shaders/wkv.spv");
    all_found &= try_load(gelu_spv_,          "shaders/gelu.spv");
    all_found &= try_load(sigmoid_spv_,       "shaders/sigmoid.spv");
    all_found &= try_load(rms_norm_spv_,      "shaders/rms_norm.spv");
    all_found &= try_load(ew_mul_spv_,        "shaders/ew_mul.spv");
    all_found &= try_load(ew_add_spv_,        "shaders/ew_add.spv");

    if (!all_found) {
        std::cout << "VulkanBackend: SPIR-V files not found, compiling shaders at runtime..." << std::endl;
        if (!compile_shaders_runtime()) {
            std::cerr << "VulkanBackend: Failed to compile shaders at runtime" << std::endl;
            return false;
        }
        // Пробуем загрузить снова после компиляции
        all_found = true;
        all_found &= try_load(matmul_small_spv_,  "shaders/matmul_small.spv");
        all_found &= try_load(matmul_medium_spv_, "shaders/matmul_medium.spv");
        all_found &= try_load(matmul_large_spv_,  "shaders/matmul_large.spv");
        all_found &= try_load(softmax_spv_,       "shaders/softmax.spv");
        all_found &= try_load(wkv_spv_,           "shaders/wkv.spv");
        all_found &= try_load(gelu_spv_,          "shaders/gelu.spv");
        all_found &= try_load(sigmoid_spv_,       "shaders/sigmoid.spv");
        all_found &= try_load(rms_norm_spv_,      "shaders/rms_norm.spv");
        all_found &= try_load(ew_mul_spv_,        "shaders/ew_mul.spv");
        all_found &= try_load(ew_add_spv_,        "shaders/ew_add.spv");
    }

    if (!all_found) {
        std::cerr << "VulkanBackend: Failed to load SPIR-V files after compilation" << std::endl;
    }
    return all_found;
}

bool VulkanBackend::compile_shaders_runtime() {
    // Создаём директорию shaders если нет
    system("mkdir -p shaders");

    // Вызываем glslc для каждого шейдера
    const char* shaders[] = {
        "matmul_small", "matmul_medium", "matmul_large",
        "softmax", "wkv", "gelu", "sigmoid", "rms_norm", "ew_mul", "ew_add",
        "matmul_fp16_large", "softmax_subgroup", "wkv_fp16",
        "rms_norm_subgroup", "gelu_fp16", "matmul_small_vec4"
    };

    for (const char* name : shaders) {
        std::string cmd = "glslc -O -fshader-stage=compute --target-env=vulkan1.2 shaders/";
        cmd += name;
        cmd += ".comp -o shaders/";
        cmd += name;
        cmd += ".spv 2>/dev/null";

        int ret = system(cmd.c_str());
        if (ret != 0) {
            // Пробуем без GCN-оптимизаций если glslc不支持 FP16
            std::string fallback = "glslc -O -fshader-stage=compute shaders/";
            fallback += name;
            fallback += ".comp -o shaders/";
            fallback += name;
            fallback += ".spv 2>/dev/null";
            system(fallback.c_str());
        }
    }
    return true;
}

// ============================================================================
// Создание шейдерных модулей
// ============================================================================
VkShaderModule VulkanBackend::create_shader_module(const std::vector<uint32_t>& spirv) const {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkShaderModule module;
    VkResult result = vkCreateShaderModule(device_, &createInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create shader module" << std::endl;
        return VK_NULL_HANDLE;
    }
    return module;
}

// ============================================================================
// Создание одного compute pipeline
// ============================================================================
VkPipeline VulkanBackend::create_compute_pipeline(VkShaderModule shader,
                                                  uint32_t local_x,
                                                  uint32_t local_y,
                                                  uint32_t local_z) const {
    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipeline_layout_;

    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create compute pipeline" << std::endl;
        return VK_NULL_HANDLE;
    }
    return pipeline;
}

// ============================================================================
// Создание всех пайплайнов
// ============================================================================
bool VulkanBackend::create_all_pipelines() {
    VkShaderModule shader_small    = create_shader_module(matmul_small_spv_);
    VkShaderModule shader_medium   = create_shader_module(matmul_medium_spv_);
    VkShaderModule shader_large    = create_shader_module(matmul_large_spv_);
    VkShaderModule shader_softmax  = create_shader_module(softmax_spv_);
    VkShaderModule shader_wkv      = create_shader_module(wkv_spv_);
    VkShaderModule shader_gelu     = create_shader_module(gelu_spv_);
    VkShaderModule shader_sigmoid  = create_shader_module(sigmoid_spv_);
    VkShaderModule shader_rms_norm = create_shader_module(rms_norm_spv_);
    VkShaderModule shader_ew_mul   = create_shader_module(ew_mul_spv_);
    VkShaderModule shader_ew_add   = create_shader_module(ew_add_spv_);

    bool ok = true;
    matmul_small_pipeline_  = create_compute_pipeline(shader_small,    4,  4,  1);  if (!matmul_small_pipeline_)  ok = false;
    matmul_medium_pipeline_ = create_compute_pipeline(shader_medium,   8,  8,  1);  if (!matmul_medium_pipeline_) ok = false;
    matmul_large_pipeline_  = create_compute_pipeline(shader_large,   16, 16,  1);  if (!matmul_large_pipeline_)  ok = false;
    softmax_pipeline_       = create_compute_pipeline(shader_softmax, 256,  1,  1);  if (!softmax_pipeline_)       ok = false;
    wkv_pipeline_           = create_compute_pipeline(shader_wkv,     256,  1,  1);  if (!wkv_pipeline_)           ok = false;
    gelu_pipeline_          = create_compute_pipeline(shader_gelu,    256,  1,  1);  if (!gelu_pipeline_)          ok = false;
    sigmoid_pipeline_       = create_compute_pipeline(shader_sigmoid, 256,  1,  1);  if (!sigmoid_pipeline_)       ok = false;
    rms_norm_pipeline_      = create_compute_pipeline(shader_rms_norm,256,  1,  1);  if (!rms_norm_pipeline_)      ok = false;
    ew_mul_pipeline_        = create_compute_pipeline(shader_ew_mul,  256,  1,  1);  if (!ew_mul_pipeline_)        ok = false;
    ew_add_pipeline_        = create_compute_pipeline(shader_ew_add,  256,  1,  1);  if (!ew_add_pipeline_)        ok = false;

    vkDestroyShaderModule(device_, shader_small,    nullptr);
    vkDestroyShaderModule(device_, shader_medium,   nullptr);
    vkDestroyShaderModule(device_, shader_large,    nullptr);
    vkDestroyShaderModule(device_, shader_softmax,  nullptr);
    vkDestroyShaderModule(device_, shader_wkv,      nullptr);
    vkDestroyShaderModule(device_, shader_gelu,     nullptr);
    vkDestroyShaderModule(device_, shader_sigmoid,  nullptr);
    vkDestroyShaderModule(device_, shader_rms_norm, nullptr);
    vkDestroyShaderModule(device_, shader_ew_mul,   nullptr);
    vkDestroyShaderModule(device_, shader_ew_add,   nullptr);

    return ok;
}

// ============================================================================
// Пул буферов
// ============================================================================
void VulkanBackend::allocate_buffer_pool() {
    VkDeviceSize poolSize = static_cast<VkDeviceSize>(config_.buffer_pool_size_mb) * 1024 * 1024;
    VkDeviceSize perBuffer = poolSize / config_.max_buffers;
    if (perBuffer < 65536) perBuffer = 65536;
    if (perBuffer > 256 * 1024 * 1024) perBuffer = 256 * 1024 * 1024;

    buffer_pool_.clear();
    for (uint32_t i = 0; i < config_.max_buffers; i++) {
        buffer_pool_.emplace_back();
        PoolEntry& entry = buffer_pool_.back();
        GPUBuffer& buf = entry.buffer;
        buf.size = perBuffer;
        buf.is_device_local = true;
        buf.is_persistent = false;
        buf.mapped = nullptr;
        entry.free = true;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = perBuffer;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buf.buffer) != VK_SUCCESS) {
            std::cerr << "VulkanBackend: Failed to allocate pool buffer " << i << std::endl;
            continue;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device_, buf.buffer, &memReqs);
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &buf.memory) != VK_SUCCESS) {
            std::cerr << "VulkanBackend: Failed to allocate pool memory " << i << std::endl;
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            buf.buffer = VK_NULL_HANDLE;
            continue;
        }
        vkBindBufferMemory(device_, buf.buffer, buf.memory, 0);
    }
}

void VulkanBackend::allocate_staging_buffer() {
    VkDeviceSize size = static_cast<VkDeviceSize>(config_.staging_buffer_size_mb) * 1024 * 1024;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &staging_buffer_.buffer) != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to create staging buffer" << std::endl;
        return;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, staging_buffer_.buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device_, &allocInfo, nullptr, &staging_buffer_.memory) != VK_SUCCESS) {
        std::cerr << "VulkanBackend: Failed to allocate staging memory" << std::endl;
        vkDestroyBuffer(device_, staging_buffer_.buffer, nullptr);
        staging_buffer_.buffer = VK_NULL_HANDLE;
        return;
    }

    vkBindBufferMemory(device_, staging_buffer_.buffer, staging_buffer_.memory, 0);
    staging_buffer_.size = size;
    staging_buffer_.is_device_local = false;
    staging_buffer_.is_persistent = false;

    vkMapMemory(device_, staging_buffer_.memory, 0, size, 0, &staging_buffer_.mapped);
    staging_offset_ = 0;
}

// ============================================================================
// Управление буферами
// ============================================================================
GPUBuffer* VulkanBackend::allocate_buffer(size_t size, bool persistent) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    VkDeviceSize perBuffer = (static_cast<VkDeviceSize>(config_.buffer_pool_size_mb) * 1024 * 1024) / config_.max_buffers;
    if (size > perBuffer) {
        buffer_pool_.emplace_back();
        PoolEntry& entry = buffer_pool_.back();
        GPUBuffer& buf = entry.buffer;
        buf.size = size;
        buf.is_device_local = true;
        buf.is_persistent = persistent;
        buf.mapped = nullptr;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buf.buffer) != VK_SUCCESS) {
            buffer_pool_.pop_back();
            return nullptr;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device_, buf.buffer, &memReqs);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &buf.memory) != VK_SUCCESS) {
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            buffer_pool_.pop_back();
            return nullptr;
        }

        vkBindBufferMemory(device_, buf.buffer, buf.memory, 0);
        entry.free = false;
        pool_total_ += size;
        pool_used_ += size;
        return &entry.buffer;
    }

    for (auto& entry : buffer_pool_) {
        if (entry.free && entry.buffer.buffer != VK_NULL_HANDLE && entry.buffer.size >= size) {
            entry.free = false;
            entry.buffer.is_persistent = persistent;
            pool_used_ += entry.buffer.size;
            return &entry.buffer;
        }
    }

    return nullptr;
}

void VulkanBackend::free_buffer(GPUBuffer* buf) {
    if (!buf) return;
    std::lock_guard<std::mutex> lock(pool_mutex_);
    VkDeviceSize perBuffer = (static_cast<VkDeviceSize>(config_.buffer_pool_size_mb) * 1024 * 1024) / config_.max_buffers;

    for (auto it = buffer_pool_.begin(); it != buffer_pool_.end(); ++it) {
        if (&it->buffer == buf) {
            if (buf->size > perBuffer) {
                vkDestroyBuffer(device_, buf->buffer, nullptr);
                vkFreeMemory(device_, buf->memory, nullptr);
                pool_total_ -= buf->size;
                pool_used_ -= buf->size;
                buffer_pool_.erase(it);
            } else {
                it->free = true;
                pool_used_ -= buf->size;
            }
            return;
        }
    }
}

// ============================================================================
// Загрузка / выгрузка данных
// ============================================================================
void VulkanBackend::upload_to_buffer(GPUBuffer* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return;
    if (size > staging_buffer_.size) {
        std::cerr << "VulkanBackend: Upload size " << size << " exceeds staging buffer " << staging_buffer_.size << std::endl;
        return;
    }

    memcpy(staging_buffer_.mapped, src, size);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResetFences(device_, 1, &batch_fence_);
    vkResetCommandBuffer(current_cmd_, 0);
    vkBeginCommandBuffer(current_cmd_, &beginInfo);
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(current_cmd_, staging_buffer_.buffer, dst->buffer, 1, &copyRegion);
    vkEndCommandBuffer(current_cmd_);
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &current_cmd_;
    vkQueueSubmit(compute_queue_, 1, &submitInfo, batch_fence_);
    vkWaitForFences(device_, 1, &batch_fence_, VK_TRUE, UINT64_MAX);
}

void VulkanBackend::download_from_buffer(void* dst, GPUBuffer* src, size_t size) {
    if (!dst || !src || size == 0) return;
    if (size > staging_buffer_.size) {
        std::cerr << "VulkanBackend: Download size " << size << " exceeds staging buffer" << std::endl;
        return;
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResetFences(device_, 1, &batch_fence_);
    vkResetCommandBuffer(current_cmd_, 0);
    vkBeginCommandBuffer(current_cmd_, &beginInfo);
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(current_cmd_, src->buffer, staging_buffer_.buffer, 1, &copyRegion);
    vkEndCommandBuffer(current_cmd_);
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &current_cmd_;
    vkQueueSubmit(compute_queue_, 1, &submitInfo, batch_fence_);
    vkWaitForFences(device_, 1, &batch_fence_, VK_TRUE, UINT64_MAX);

    memcpy(dst, staging_buffer_.mapped, size);
    vkResetCommandBuffer(current_cmd_, 0);
}
void VulkanBackend::upload_to_buffer_batched(GPUBuffer* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return;
    if (size > staging_buffer_.size) return;
    ensure_batch_open();
    memcpy(staging_buffer_.mapped, src, size);
    VkBufferCopy r = {}; r.size = size;
    vkCmdCopyBuffer(current_cmd_, staging_buffer_.buffer, dst->buffer, 1, &r);
    VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.buffer = dst->buffer; b.size = size;
    vkCmdPipelineBarrier(current_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
}

void VulkanBackend::download_from_buffer_batched(void* dst, GPUBuffer* src, size_t size) {
    if (!dst || !src || size == 0) return;
    if (staging_used_ + size > staging_buffer_.size) {
        std::cerr << "VulkanBackend: FATAL staging overflow " << (staging_used_ + size)
                  << " > " << staging_buffer_.size << std::endl;
        std::abort();
    }
    ensure_batch_open();
    VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.buffer = src->buffer; b.size = size;
    vkCmdPipelineBarrier(current_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
    VkBufferCopy r = {}; r.dstOffset = staging_used_; r.size = size;
    vkCmdCopyBuffer(current_cmd_, src->buffer, staging_buffer_.buffer, 1, &r);
    PendingDownload pd;
    pd.dst = nullptr;
    pd.size = size;
    pd.offset = staging_used_;
    pd.data.resize(size);
    pending_downloads_.push_back(std::move(pd));
    staging_used_ += size;
}
// ============================================================================
// Батчинг
// ============================================================================
void VulkanBackend::ensure_batch_open() {
    if (batch_open_) return;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(current_cmd_, &beginInfo);
    batch_open_ = true;
}

void VulkanBackend::begin_batch() {
    if (batch_open_) {
        end_batch();
    }
    pending_downloads_.clear();
    staging_used_ = 0;
    ensure_batch_open();
}

void VulkanBackend::end_batch() {
    if (!batch_open_) return;

    vkEndCommandBuffer(current_cmd_);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &current_cmd_;

    vkResetFences(device_, 1, &batch_fence_);
    VkResult result = vkQueueSubmit(compute_queue_, 1, &submitInfo, batch_fence_);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: vkQueueSubmit failed" << std::endl;
        return;
    }

    result = vkWaitForFences(device_, 1, &batch_fence_, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cerr << "VulkanBackend: vkWaitForFences failed" << std::endl;
        return;
    }

    for (auto& dl : pending_downloads_) {
        if (!dl.data.empty()) {
            memcpy(dl.data.data(), (char*)staging_buffer_.mapped + dl.offset, dl.size);
        } else if (dl.dst) {
            memcpy(dl.dst, (char*)staging_buffer_.mapped + dl.offset, dl.size);
        }
    }
    pending_downloads_.clear();
    staging_used_ = 0;

    vkResetCommandBuffer(current_cmd_, 0);
    batch_open_ = false;
}

// ============================================================================
// Выделение дескрипторного набора
// ============================================================================
DescriptorSetSlot* VulkanBackend::allocate_descriptor_set(const std::vector<GPUBuffer*>& buffers) {
    std::lock_guard<std::mutex> lock(desc_mutex_);

    for (auto& slot : desc_slots_) {
        if (!slot.in_use) {
            slot.in_use = true;

            for (size_t i = 0; i < buffers.size() && i < 6; i++) {
                slot.buffer_infos[i] = {};
                slot.buffer_infos[i].buffer = buffers[i]->buffer;
                slot.buffer_infos[i].offset = 0;
                slot.buffer_infos[i].range = VK_WHOLE_SIZE;

                VkWriteDescriptorSet write = {};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = slot.set;
                write.dstBinding = static_cast<uint32_t>(i);
                write.dstArrayElement = 0;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &slot.buffer_infos[i];

                vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
            }

            return &slot;
        }
    }

    return nullptr;
}

void VulkanBackend::free_descriptor_set(DescriptorSetSlot* slot) {
    if (slot) slot->in_use = false;
}

// ============================================================================
// MATMUL
// ============================================================================
void VulkanBackend::matmul(GPUBuffer* A, GPUBuffer* B, GPUBuffer* C,
                           uint32_t M, uint32_t K, uint32_t N,
                           bool transpose_A, bool transpose_B) {
    ensure_batch_open();

    VkPipeline pipeline;
    uint32_t wg_x, wg_y;

    size_t total_ops = static_cast<size_t>(M) * K + static_cast<size_t>(K) * N;

    if (total_ops < 65536) {
        pipeline = matmul_small_pipeline_;
        wg_x = (M + 3) / 4;
        wg_y = (N + 3) / 4;
    } else if (total_ops < 16777216) {
        pipeline = matmul_medium_pipeline_;
        wg_x = (M + 7) / 8;
        wg_y = (N + 7) / 8;
    } else {
        pipeline = matmul_large_pipeline_;
        wg_x = (M + 15) / 16;
        wg_y = (N + 15) / 16;
    }

    std::vector<GPUBuffer*> bufs = {A, B, C};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[5] = {M, K, N, transpose_A ? 1u : 0u, transpose_B ? 1u : 0u};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    vkCmdDispatch(current_cmd_, wg_x, wg_y, 1);

    free_descriptor_set(slot);
}

// ============================================================================
// SOFTMAX
// ============================================================================
void VulkanBackend::softmax(GPUBuffer* input, GPUBuffer* output,
                            uint32_t rows, uint32_t cols) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {input, output};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, softmax_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[2] = {rows, cols};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    vkCmdDispatch(current_cmd_, rows, 1, 1);

    free_descriptor_set(slot);
}

// ============================================================================
// WKV
// ============================================================================
void VulkanBackend::wkv(GPUBuffer* k, GPUBuffer* v, GPUBuffer* state,
                        GPUBuffer* out, GPUBuffer* decay, GPUBuffer* first,
                        uint32_t n_embd) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {k, v, state, out, decay, first};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, wkv_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[1] = {n_embd};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    uint32_t groups = (n_embd + 255) / 256;
    vkCmdDispatch(current_cmd_, groups, 1, 1);

    free_descriptor_set(slot);
}

// ============================================================================
// GELU, SIGMOID, RMS_NORM, EW_MUL, EW_ADD
// ============================================================================
void VulkanBackend::gelu(GPUBuffer* input, GPUBuffer* output, uint32_t n) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {input, output};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, gelu_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[1] = {n};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    uint32_t groups = (n + 255) / 256;
    vkCmdDispatch(current_cmd_, groups, 1, 1);

    free_descriptor_set(slot);
}

void VulkanBackend::sigmoid(GPUBuffer* input, GPUBuffer* output, uint32_t n) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {input, output};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, sigmoid_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[1] = {n};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    uint32_t groups = (n + 255) / 256;
    vkCmdDispatch(current_cmd_, groups, 1, 1);

    free_descriptor_set(slot);
}

void VulkanBackend::rms_norm(GPUBuffer* input, GPUBuffer* output, uint32_t n) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {input, output};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, rms_norm_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[1] = {n};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    uint32_t groups = (n + 255) / 256;
    vkCmdDispatch(current_cmd_, groups, 1, 1);

    free_descriptor_set(slot);
}

void VulkanBackend::element_wise_mul(GPUBuffer* a, GPUBuffer* b, GPUBuffer* c, uint32_t n) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {a, b, c};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, ew_mul_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[1] = {n};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    uint32_t groups = (n + 255) / 256;
    vkCmdDispatch(current_cmd_, groups, 1, 1);

    free_descriptor_set(slot);
}

void VulkanBackend::element_wise_add(GPUBuffer* a, GPUBuffer* b, GPUBuffer* c, uint32_t n) {
    ensure_batch_open();

    std::vector<GPUBuffer*> bufs = {a, b, c};
    DescriptorSetSlot* slot = allocate_descriptor_set(bufs);
    if (!slot) return;

    vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, ew_add_pipeline_);
    vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &slot->set, 0, nullptr);

    uint32_t push[1] = {n};
    vkCmdPushConstants(current_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    uint32_t groups = (n + 255) / 256;
    vkCmdDispatch(current_cmd_, groups, 1, 1);

    free_descriptor_set(slot);
}

// ============================================================================
// Вспомогательные
// ============================================================================
uint32_t VulkanBackend::find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return 0;
}

void VulkanBackend::print_device_info() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);

    std::cout << "Vulkan Device: " << props.deviceName << "\n";
    std::cout << "  API Version: " << VK_VERSION_MAJOR(props.apiVersion) << "."
              << VK_VERSION_MINOR(props.apiVersion) << "."
              << VK_VERSION_PATCH(props.apiVersion) << "\n";
    std::cout << "  Driver Version: " << props.driverVersion << "\n";
    std::cout << "  Device Type: " <<
        (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU" :
         props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "Integrated GPU" :
         props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "CPU" : "Other") << "\n";
    std::cout << "  Max Compute Work Group Size: ["
              << props.limits.maxComputeWorkGroupSize[0] << ", "
              << props.limits.maxComputeWorkGroupSize[1] << ", "
              << props.limits.maxComputeWorkGroupSize[2] << "]\n";
    std::cout << "  Max Compute Work Group Count: ["
              << props.limits.maxComputeWorkGroupCount[0] << ", "
              << props.limits.maxComputeWorkGroupCount[1] << ", "
              << props.limits.maxComputeWorkGroupCount[2] << "]\n";
    std::cout << "  Max Compute Work Group Invocations: "
              << props.limits.maxComputeWorkGroupInvocations << "\n";
    std::cout << "  Max Shared Memory: " << props.limits.maxComputeSharedMemorySize / 1024 << " KB\n";
}

size_t VulkanBackend::get_memory_usage() const {
    return pool_used_ + staging_buffer_.size;
}

} // namespace uzagpt
