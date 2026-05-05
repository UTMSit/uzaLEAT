// файл: vulkan_backend.cpp
#include "vulkan_backend.hpp"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <shaderc/shaderc.hpp>
#include <iostream>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace uzagpt {

static const std::string matmul_glsl = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer B { float b[]; };
layout(set = 0, binding = 2) buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint K;
    uint N;
};

shared float Asub[16][17];
shared float Bsub[16][17];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;

    float sum = 0.0;
    uint tiles = (K + 15) / 16;

    for (uint t = 0; t < tiles; t++) {
        uint a_idx = row * K + t * 16 + gl_LocalInvocationID.x;
        uint b_idx = (t * 16 + gl_LocalInvocationID.y) * N + col;

        Asub[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
            (row < M && t * 16 + gl_LocalInvocationID.x < K) ? a[a_idx] : 0.0;
        Bsub[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
            (t * 16 + gl_LocalInvocationID.y < K && col < N) ? b[b_idx] : 0.0;

        barrier();

        for (uint k = 0; k < 16; k++) {
            sum += Asub[gl_LocalInvocationID.y][k] * Bsub[k][gl_LocalInvocationID.x];
        }

        barrier();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
)";

static const std::string wkv_glsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer K { float k[]; };
layout(set = 0, binding = 1) buffer V { float v[]; };
layout(set = 0, binding = 2) buffer State { float state[]; };
layout(set = 0, binding = 3) buffer Out { float out_val[]; };
layout(set = 0, binding = 4) buffer Decay { float decay[]; };
layout(set = 0, binding = 5) buffer First { float first[]; };

layout(push_constant) uniform Params {
    uint n_embd;
    uint pos;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n_embd) return;

    float k_t = k[i];
    float v_t = v[i];
    float d = exp(decay[i]);
    float f = first[i];
    float s = state[i];

    float log_efk = f + k_t;
    float max_val = max(log_efk, 0.0);
    float exp_efk = exp(log_efk - max_val);
    float exp_d = d * exp(-max_val);

    float num = exp_efk * v_t + exp_d * s;
    float den = exp_efk + exp_d;

    float wkv = num / (den + 1e-8);
    out_val[i] = wkv;
    state[i] = wkv;
}
)";

static const std::string softmax_glsl = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InputBuffer { float input_data[]; };
layout(set = 0, binding = 1) buffer OutputBuffer { float output_data[]; };

layout(push_constant) uniform Params {
    uint rows;
    uint cols;
};

shared float shared_max[256];
shared float shared_sum[256];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_LocalInvocationID.x;

    if (row >= rows) return;

    float max_val = -1e30;
    for (uint i = col; i < cols; i += gl_WorkGroupSize.x) {
        max_val = max(max_val, input_data[row * cols + i]);
    }
    shared_max[col] = max_val;
    barrier();

    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride /= 2) {
        if (col < stride) {
            shared_max[col] = max(shared_max[col], shared_max[col + stride]);
        }
        barrier();
    }
    max_val = shared_max[0];

    float sum = 0.0;
    for (uint i = col; i < cols; i += gl_WorkGroupSize.x) {
        float val = exp(input_data[row * cols + i] - max_val);
        output_data[row * cols + i] = val;
        sum += val;
    }
    shared_sum[col] = sum;
    barrier();

    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride /= 2) {
        if (col < stride) {
            shared_sum[col] += shared_sum[col + stride];
        }
        barrier();
    }
    sum = shared_sum[0];

    float inv_sum = 1.0 / (sum + 1e-8);
    for (uint i = col; i < cols; i += gl_WorkGroupSize.x) {
        output_data[row * cols + i] *= inv_sum;
    }
}
)";

#define VK_CHECK(f) { VkResult res = (f); if (res != VK_SUCCESS) { std::cerr << "Vulkan error: " << res << " at " << __FILE__ << ":" << __LINE__ << "\n"; } }

VulkanBackend::VulkanBackend() = default;
VulkanBackend::~VulkanBackend() { shutdown(); }

bool VulkanBackend::compile_glsl_to_spirv(const std::string& glsl_source,
                                          const std::string& entry_point,
                                          std::vector<uint32_t>& spirv) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetSourceLanguage(shaderc_source_language_glsl);

    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
        glsl_source, shaderc_compute_shader, "shader", entry_point.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << "Shader compilation error: " << result.GetErrorMessage() << std::endl;
        return false;
    }

    spirv.assign(result.cbegin(), result.cend());
    return true;
}

bool VulkanBackend::create_shader_module(const std::vector<uint32_t>& spirv, VkShaderModule* module) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();
    return vkCreateShaderModule(device_, &createInfo, nullptr, module) == VK_SUCCESS;
}

bool VulkanBackend::create_compute_pipeline(VkShaderModule shader, VkPipelineLayout* layout,
                                            VkPipeline* pipeline, uint32_t /*local_x*/,
                                            uint32_t /*local_y*/, uint32_t /*local_z*/) {
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 3;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptor_set_layout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, layout));

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shader;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = *layout;

    return vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pipeline) == VK_SUCCESS;
}

bool VulkanBackend::init() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "UzaLEAT";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "UzaLEAT";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    VK_CHECK(vkCreateInstance(&instInfo, nullptr, &instance_));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "Vulkan: No physical devices found\n";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    physical_device_ = devices[0];
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = dev;
            break;
        }
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount, queueFamilies.data());

    queue_family_index_ = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_index_ = i;
            break;
        }
    }
    if (queue_family_index_ == UINT32_MAX) {
        std::cerr << "Vulkan: No compute queue family found\n";
        return false;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queue_family_index_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.queueCreateInfoCount = 1;

    VK_CHECK(vkCreateDevice(physical_device_, &deviceInfo, nullptr, &device_));

    vkGetDeviceQueue(device_, queue_family_index_, 0, &compute_queue_);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queue_family_index_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &command_pool_));

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = command_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, &command_buffer_));

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &fence_));

    const uint32_t max_sets = 65536;
    const uint32_t max_descriptors = max_sets * 6;

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = max_descriptors;

    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;
    descPoolInfo.maxSets = max_sets;

    VK_CHECK(vkCreateDescriptorPool(device_, &descPoolInfo, nullptr, &descriptor_pool_));

    std::vector<VkDescriptorSetLayoutBinding> bindings(6);
    for (uint32_t i = 0; i < 6; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptor_set_layout_));

    std::vector<uint32_t> spirv;
    if (!compile_glsl_to_spirv(matmul_glsl, "main", spirv) ||
        !create_shader_module(spirv, &matmul_shader_)) {
        std::cerr << "Vulkan: Failed to compile/create matmul shader\n";
        return false;
    }
    if (!compile_glsl_to_spirv(wkv_glsl, "main", spirv) ||
        !create_shader_module(spirv, &wkv_shader_)) {
        std::cerr << "Vulkan: Failed to compile/create wkv shader\n";
        return false;
    }
    if (!compile_glsl_to_spirv(softmax_glsl, "main", spirv) ||
        !create_shader_module(spirv, &softmax_shader_)) {
        std::cerr << "Vulkan: Failed to compile/create softmax shader\n";
        return false;
    }

    if (!create_compute_pipeline(matmul_shader_, &matmul_layout_, &matmul_pipeline_, 16, 16, 1)) {
        std::cerr << "Vulkan: Failed to create matmul pipeline\n";
        return false;
    }
    if (!create_compute_pipeline(wkv_shader_, &wkv_layout_, &wkv_pipeline_, 256, 1, 1)) {
        std::cerr << "Vulkan: Failed to create wkv pipeline\n";
        return false;
    }
    if (!create_compute_pipeline(softmax_shader_, &softmax_layout_, &softmax_pipeline_, 256, 1, 1)) {
        std::cerr << "Vulkan: Failed to create softmax pipeline\n";
        return false;
    }

    std::cout << "Vulkan backend initialized successfully\n";
    print_device_info();
    return true;
}

void VulkanBackend::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        for (auto& buf : buffers_) {
            if (buf->buffer) vkDestroyBuffer(device_, buf->buffer, nullptr);
            if (buf->memory) vkFreeMemory(device_, buf->memory, nullptr);
        }
        buffers_.clear();
        ptr_to_buffer_.clear();

        if (matmul_pipeline_) vkDestroyPipeline(device_, matmul_pipeline_, nullptr);
        if (matmul_layout_)  vkDestroyPipelineLayout(device_, matmul_layout_, nullptr);
        if (wkv_pipeline_)   vkDestroyPipeline(device_, wkv_pipeline_, nullptr);
        if (wkv_layout_)     vkDestroyPipelineLayout(device_, wkv_layout_, nullptr);
        if (softmax_pipeline_) vkDestroyPipeline(device_, softmax_pipeline_, nullptr);
        if (softmax_layout_) vkDestroyPipelineLayout(device_, softmax_layout_, nullptr);
        if (matmul_shader_)  vkDestroyShaderModule(device_, matmul_shader_, nullptr);
        if (wkv_shader_)     vkDestroyShaderModule(device_, wkv_shader_, nullptr);
        if (softmax_shader_) vkDestroyShaderModule(device_, softmax_shader_, nullptr);
        if (descriptor_set_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (fence_) vkDestroyFence(device_, fence_, nullptr);
        if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void* VulkanBackend::allocate_buffer(size_t size) {
    if (!is_ready()) return nullptr;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, buffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        std::cerr << "Vulkan: No suitable memory type found\n";
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory memory;
    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));

    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descriptor_pool_;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descriptor_set_layout_;

    VkDescriptorSet descSet;
    VkResult res = vkAllocateDescriptorSets(device_, &descAllocInfo, &descSet);
    if (res != VK_SUCCESS) {
        std::cerr << "Vulkan: Failed to allocate descriptor set (result: " << res << ")\n";
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    auto gpuBuf = std::make_unique<GPUBuffer>();
    gpuBuf->buffer = buffer;
    gpuBuf->memory = memory;
    gpuBuf->descriptor_set = descSet;
    gpuBuf->size = size;

    void* ptr = gpuBuf.get();
    ptr_to_buffer_[ptr] = gpuBuf.get();
    buffers_.push_back(std::move(gpuBuf));

    return ptr;
}

void VulkanBackend::free_buffer(void* gpu_ptr) {
    auto it = ptr_to_buffer_.find(gpu_ptr);
    if (it == ptr_to_buffer_.end()) return;
    GPUBuffer* buf = it->second;
    vkDestroyBuffer(device_, buf->buffer, nullptr);
    vkFreeMemory(device_, buf->memory, nullptr);
    ptr_to_buffer_.erase(it);
    for (auto iter = buffers_.begin(); iter != buffers_.end(); ++iter) {
        if (iter->get() == buf) {
            buffers_.erase(iter);
            break;
        }
    }
}

void VulkanBackend::upload_data(void* gpu_ptr, const void* cpu_data, size_t size) {
    auto it = ptr_to_buffer_.find(gpu_ptr);
    if (it == ptr_to_buffer_.end()) return;
    GPUBuffer* buf = it->second;

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = size;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer;
    VK_CHECK(vkCreateBuffer(device_, &stagingInfo, nullptr, &stagingBuffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }

    VkDeviceMemory stagingMemory;
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &stagingMemory));
    VK_CHECK(vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));

    void* mapped;
    VK_CHECK(vkMapMemory(device_, stagingMemory, 0, size, 0, &mapped));
    memcpy(mapped, cpu_data, size);
    vkUnmapMemory(device_, stagingMemory);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(command_buffer_, &beginInfo));
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(command_buffer_, stagingBuffer, buf->buffer, 1, &copyRegion);
    VK_CHECK(vkEndCommandBuffer(command_buffer_));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(compute_queue_));

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingMemory, nullptr);
    VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));
}

void VulkanBackend::download_data(void* cpu_data, const void* gpu_ptr, size_t size) {
    auto it = ptr_to_buffer_.find(const_cast<void*>(gpu_ptr));
    if (it == ptr_to_buffer_.end()) return;
    GPUBuffer* buf = it->second;

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = size;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer;
    VK_CHECK(vkCreateBuffer(device_, &stagingInfo, nullptr, &stagingBuffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }

    VkDeviceMemory stagingMemory;
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &stagingMemory));
    VK_CHECK(vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(command_buffer_, &beginInfo));
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(command_buffer_, buf->buffer, stagingBuffer, 1, &copyRegion);
    VK_CHECK(vkEndCommandBuffer(command_buffer_));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(compute_queue_));

    void* mapped;
    VK_CHECK(vkMapMemory(device_, stagingMemory, 0, size, 0, &mapped));
    memcpy(cpu_data, mapped, size);
    vkUnmapMemory(device_, stagingMemory);

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingMemory, nullptr);
    VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));
}

void VulkanBackend::matmul(void* A, void* B, void* C, size_t M, size_t K, size_t N) {
    auto itA = ptr_to_buffer_.find(A);
    auto itB = ptr_to_buffer_.find(B);
    auto itC = ptr_to_buffer_.find(C);
    if (itA == ptr_to_buffer_.end() || itB == ptr_to_buffer_.end() || itC == ptr_to_buffer_.end())
        return;

    VkDescriptorSet sets[3] = {
        itA->second->descriptor_set,
        itB->second->descriptor_set,
        itC->second->descriptor_set
    };

    for (int i = 0; i < 3; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = (i == 0) ? itA->second->buffer :
                           (i == 1) ? itB->second->buffer :
                                      itC->second->buffer;
        bufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = sets[i];
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(command_buffer_, &beginInfo));
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, matmul_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            matmul_layout_, 0, 3, sets, 0, nullptr);

    struct { uint32_t M; uint32_t K; uint32_t N; } params = {
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(N)
    };
    vkCmdPushConstants(command_buffer_, matmul_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(params), &params);

    uint32_t groupsX = (M + 15) / 16;
    uint32_t groupsY = (N + 15) / 16;
    vkCmdDispatch(command_buffer_, groupsX, groupsY, 1);
    VK_CHECK(vkEndCommandBuffer(command_buffer_));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submitInfo, fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device_, 1, &fence_));
    VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));
}

void VulkanBackend::rwkv_wkv(void* k, void* v, void* state, void* out,
                             void* decay, void* first, size_t n_embd) {
    auto itK = ptr_to_buffer_.find(k);
    auto itV = ptr_to_buffer_.find(v);
    auto itState = ptr_to_buffer_.find(state);
    auto itOut = ptr_to_buffer_.find(out);
    auto itDecay = ptr_to_buffer_.find(decay);
    auto itFirst = ptr_to_buffer_.find(first);

    if (itK == ptr_to_buffer_.end() || itV == ptr_to_buffer_.end() ||
        itState == ptr_to_buffer_.end() || itOut == ptr_to_buffer_.end() ||
        itDecay == ptr_to_buffer_.end() || itFirst == ptr_to_buffer_.end())
        return;

    VkDescriptorSet sets[6] = {
        itK->second->descriptor_set,
        itV->second->descriptor_set,
        itState->second->descriptor_set,
        itOut->second->descriptor_set,
        itDecay->second->descriptor_set,
        itFirst->second->descriptor_set
    };

    for (int i = 0; i < 6; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        switch (i) {
            case 0: bufferInfo.buffer = itK->second->buffer; break;
            case 1: bufferInfo.buffer = itV->second->buffer; break;
            case 2: bufferInfo.buffer = itState->second->buffer; break;
            case 3: bufferInfo.buffer = itOut->second->buffer; break;
            case 4: bufferInfo.buffer = itDecay->second->buffer; break;
            case 5: bufferInfo.buffer = itFirst->second->buffer; break;
        }
        bufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = sets[i];
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(command_buffer_, &beginInfo));
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, wkv_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            wkv_layout_, 0, 6, sets, 0, nullptr);

    struct { uint32_t n_embd; uint32_t pos; } params = {
        static_cast<uint32_t>(n_embd), 0
    };
    vkCmdPushConstants(command_buffer_, wkv_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(params), &params);

    uint32_t groups = (n_embd + 255) / 256;
    vkCmdDispatch(command_buffer_, groups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(command_buffer_));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submitInfo, fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device_, 1, &fence_));
    VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));
}

void VulkanBackend::softmax(void* input, void* output, size_t rows, size_t cols) {
    auto itIn = ptr_to_buffer_.find(input);
    auto itOut = ptr_to_buffer_.find(output);
    if (itIn == ptr_to_buffer_.end() || itOut == ptr_to_buffer_.end())
        return;

    VkDescriptorSet sets[2] = {
        itIn->second->descriptor_set,
        itOut->second->descriptor_set
    };

    for (int i = 0; i < 2; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = (i == 0) ? itIn->second->buffer : itOut->second->buffer;
        bufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = sets[i];
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(command_buffer_, &beginInfo));
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, softmax_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            softmax_layout_, 0, 2, sets, 0, nullptr);

    struct { uint32_t rows; uint32_t cols; } params = {
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(cols)
    };
    vkCmdPushConstants(command_buffer_, softmax_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(params), &params);

    vkCmdDispatch(command_buffer_, rows, 1, 1);
    VK_CHECK(vkEndCommandBuffer(command_buffer_));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer_;

    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submitInfo, fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device_, 1, &fence_));
    VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));
}

void VulkanBackend::flush() {
    vkQueueWaitIdle(compute_queue_);
}

void VulkanBackend::wait() {
    if (fence_ != VK_NULL_HANDLE) {
        VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device_, 1, &fence_));
    }
}

size_t VulkanBackend::get_memory_usage() const {
    size_t total = 0;
    for (const auto& buf : buffers_) {
        total += buf->size;
    }
    return total;
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
}

} // namespace uzagpt
