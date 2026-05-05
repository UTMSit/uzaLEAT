// файл: vulkan_backend.hpp
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <iostream>
#include <cstdint>

namespace uzagpt {

struct GPUBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    size_t size = 0;
};

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();

    bool init();
    void shutdown();

    void* allocate_buffer(size_t size);
    void  free_buffer(void* gpu_ptr);
    void  upload_data(void* gpu_ptr, const void* cpu_data, size_t size);
    void  download_data(void* cpu_data, const void* gpu_ptr, size_t size);

    void matmul(void* A, void* B, void* C, size_t M, size_t K, size_t N);
    void rwkv_wkv(void* k, void* v, void* state, void* out,
                  void* decay, void* first, size_t n_embd);
    void softmax(void* input, void* output, size_t rows, size_t cols);

    void flush();
    void wait();

    bool is_ready() const { return device_ != VK_NULL_HANDLE; }
    size_t get_memory_usage() const;
    void print_device_info() const;

private:
    VkInstance       instance_        = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice         device_          = VK_NULL_HANDLE;
    VkQueue          compute_queue_   = VK_NULL_HANDLE;
    VkCommandPool    command_pool_    = VK_NULL_HANDLE;
    VkCommandBuffer  command_buffer_  = VK_NULL_HANDLE;
    VkFence          fence_           = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;

    uint32_t queue_family_index_ = UINT32_MAX;

    VkPipelineLayout matmul_layout_   = VK_NULL_HANDLE;
    VkPipeline       matmul_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout wkv_layout_      = VK_NULL_HANDLE;
    VkPipeline       wkv_pipeline_    = VK_NULL_HANDLE;
    VkPipelineLayout softmax_layout_  = VK_NULL_HANDLE;
    VkPipeline       softmax_pipeline_= VK_NULL_HANDLE;

    VkShaderModule matmul_shader_   = VK_NULL_HANDLE;
    VkShaderModule wkv_shader_      = VK_NULL_HANDLE;
    VkShaderModule softmax_shader_  = VK_NULL_HANDLE;

    std::vector<std::unique_ptr<GPUBuffer>> buffers_;
    std::unordered_map<void*, GPUBuffer*> ptr_to_buffer_;

    bool compile_glsl_to_spirv(const std::string& glsl_source,
                               const std::string& entry_point,
                               std::vector<uint32_t>& spirv);
    bool create_shader_module(const std::vector<uint32_t>& spirv, VkShaderModule* module);
    bool create_compute_pipeline(VkShaderModule shader, VkPipelineLayout* layout,
                                 VkPipeline* pipeline, uint32_t local_x = 256,
                                 uint32_t local_y = 1, uint32_t local_z = 1);
};

} // namespace uzagpt
