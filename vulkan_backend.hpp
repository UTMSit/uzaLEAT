// ============================================================================
// vulkan_backend.hpp — Production GPU Backend для uzaLEAT
// ============================================================================
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <list>

namespace uzagpt {

struct GPUConfig {
    bool enable_async_compute = true;
    bool enable_async_transfer = true;
    bool enable_fp16_math = true;
    bool enable_push_descriptors = false;
    uint32_t buffer_pool_size_mb = 512;
    uint32_t max_buffers = 4096;
    uint32_t staging_buffer_size_mb = 64;
};

struct GPUBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool is_device_local = true;
    bool is_persistent = false;
};

struct DescriptorSetSlot {
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    bool in_use = false;
};

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();

    bool init(const GPUConfig& cfg = GPUConfig());
    void shutdown();
    bool is_ready() const { return device_ != VK_NULL_HANDLE; }

    GPUBuffer* allocate_buffer(size_t size, bool persistent = false);
    void free_buffer(GPUBuffer* buf);
    void upload_to_buffer(GPUBuffer* dst, const void* src, size_t size);
    void download_from_buffer(void* dst, GPUBuffer* src, size_t size);
    void upload_to_buffer_batched(GPUBuffer* dst, const void* src, size_t size);
    void download_from_buffer_batched(void* dst, GPUBuffer* src, size_t size);

    void begin_batch();
    void end_batch();
    void flush();

    void matmul(GPUBuffer* A, GPUBuffer* B, GPUBuffer* C,
                uint32_t M, uint32_t K, uint32_t N,
                bool transpose_A = false, bool transpose_B = false);

    void softmax(GPUBuffer* input, GPUBuffer* output,
                 uint32_t rows, uint32_t cols);

    void wkv(GPUBuffer* k, GPUBuffer* v, GPUBuffer* state,
             GPUBuffer* out, GPUBuffer* decay, GPUBuffer* first,
             uint32_t n_embd);

    void gelu(GPUBuffer* input, GPUBuffer* output, uint32_t n);
    void sigmoid(GPUBuffer* input, GPUBuffer* output, uint32_t n);
    void rms_norm(GPUBuffer* input, GPUBuffer* output, uint32_t n);
    void element_wise_mul(GPUBuffer* a, GPUBuffer* b, GPUBuffer* c, uint32_t n);
    void element_wise_add(GPUBuffer* a, GPUBuffer* b, GPUBuffer* c, uint32_t n);

    void print_device_info() const;
    size_t get_memory_usage() const;
    uint32_t get_max_workgroup_size() const { return max_workgroup_size_; }
    uint32_t get_max_workgroup_invocations() const { return max_workgroup_invocations_; }

    VkDevice get_device() const { return device_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    VkCommandBuffer get_current_cmd() const { return current_cmd_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkQueue transfer_queue_ = VK_NULL_HANDLE;
    uint32_t compute_family_ = 0;
    uint32_t transfer_family_ = 0;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer current_cmd_ = VK_NULL_HANDLE;
    VkFence batch_fence_ = VK_NULL_HANDLE;
    bool batch_open_ = false;

    VkCommandPool transfer_cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer transfer_cmd_ = VK_NULL_HANDLE;
    VkFence transfer_fence_ = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    std::vector<DescriptorSetSlot> desc_slots_;
    std::mutex desc_mutex_;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline matmul_small_pipeline_ = VK_NULL_HANDLE;
    VkPipeline matmul_medium_pipeline_ = VK_NULL_HANDLE;
    VkPipeline matmul_large_pipeline_ = VK_NULL_HANDLE;
    VkPipeline softmax_pipeline_ = VK_NULL_HANDLE;
    VkPipeline wkv_pipeline_ = VK_NULL_HANDLE;
    VkPipeline gelu_pipeline_ = VK_NULL_HANDLE;
    VkPipeline sigmoid_pipeline_ = VK_NULL_HANDLE;
    VkPipeline rms_norm_pipeline_ = VK_NULL_HANDLE;
    VkPipeline ew_mul_pipeline_ = VK_NULL_HANDLE;
    VkPipeline ew_add_pipeline_ = VK_NULL_HANDLE;

    struct PoolEntry {
        GPUBuffer buffer;
        bool free = true;
    };
    std::list<PoolEntry> buffer_pool_;
    std::mutex pool_mutex_;
    VkDeviceSize pool_total_ = 0;
    VkDeviceSize pool_used_ = 0;

    GPUBuffer staging_buffer_;
    struct PendingDownload {
        void* dst;
        size_t size;
        std::vector<char> data;  // владеет данными
    };
    std::vector<PendingDownload> pending_downloads_;
    size_t staging_used_ = 0;
    size_t staging_offset_ = 0;

    uint32_t max_workgroup_size_ = 256;
    uint32_t max_workgroup_invocations_ = 1024;
    uint32_t max_shared_memory_ = 32768;
    bool has_fp16_ = false;
    bool has_async_compute_ = false;
    VkPhysicalDeviceSubgroupProperties subgroup_props_{};

    GPUConfig config_;

    std::vector<uint32_t> matmul_small_spv_;
    std::vector<uint32_t> matmul_medium_spv_;
    std::vector<uint32_t> matmul_large_spv_;
    std::vector<uint32_t> softmax_spv_;
    std::vector<uint32_t> wkv_spv_;
    std::vector<uint32_t> gelu_spv_;
    std::vector<uint32_t> sigmoid_spv_;
    std::vector<uint32_t> rms_norm_spv_;
    std::vector<uint32_t> ew_mul_spv_;
    std::vector<uint32_t> ew_add_spv_;

    void create_instance();
    void pick_physical_device();
    void create_logical_device();
    void create_queues();
    void create_command_pools();
    void create_descriptor_layout();
    void create_pipeline_layout();
    bool load_shaders_from_files();
    bool compile_shaders_runtime();
    bool create_all_pipelines();
    void allocate_buffer_pool();
    void allocate_staging_buffer();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    VkShaderModule create_shader_module(const std::vector<uint32_t>& spirv) const;
    VkPipeline create_compute_pipeline(VkShaderModule shader,
                                       uint32_t local_x = 256,
                                       uint32_t local_y = 1,
                                       uint32_t local_z = 1) const;
    DescriptorSetSlot* allocate_descriptor_set(const std::vector<GPUBuffer*>& buffers);
    void free_descriptor_set(DescriptorSetSlot* slot);
    void ensure_batch_open();
};

} // namespace uzagpt
