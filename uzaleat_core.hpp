#ifndef UZALEAT_CORE_HPP
#define UZALEAT_CORE_HPP

#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <random>
#include <fstream>
#include <atomic>
#include <unordered_map>
#include <functional>
#include "gutr_vm.hpp"
#include <dlfcn.h>

#ifdef UZALEAT_USE_VULKAN
#include "vulkan_backend.hpp"
#endif

namespace uzaleat {

struct CoreConfig {
    std::string plugin_path;
    std::string model_so_path;
    std::string data_path;
    std::string tokenizer_path;
    std::string model_path;
    bool train_mode = false;
    bool chat_mode = false;

    size_t hidden_size = 768;
    size_t num_layers = 12;
    size_t context_size = 1024;
    size_t vocab_size = 50257;

    int epochs = 3;
    float learning_rate = 0.0001f;
    int batch_size = 8;
    int num_threads = 4;
    int shuffle_buffer = 10000;
    int sample_every = 50;
    int max_tokens_sample = 50;
    float temperature = 0.8f;
    float top_p = 0.9f;

    int tt_rank = 64;
    int proj_rank = 64;
    int update_interval = 10000;
    int num_experts = 6;
    int window_size = 4096;

    bool use_gpu = false;
};

struct ModelSO {
    void* handle = nullptr;
    void (*init)(int,int,int,int,int,int,int);
    float (*train_step)(const int*,const int*,int,float);
    int (*generate)(const int*,int,int*,int,float,float);
    void (*save)(const char*);
    void (*load)(const char*);
};

class StreamingDataset {
public:
    StreamingDataset(const std::string& path, int shuffle_buffer);
    bool next(std::pair<std::string, std::string>& example);
    void reset();

private:
    std::vector<std::string> files_;
    size_t current_file_idx_ = 0;
    std::ifstream current_file_;
    std::string current_input_key_;
    std::string current_output_key_;
    bool in_array_ = false;
    int shuffle_buffer_;
    std::queue<std::pair<std::string, std::string>> shuffle_pool_;
    std::mt19937 rng_;
    std::mutex mutex_;

    bool open_next_file();
    bool detect_keys_for_current_file();
    std::pair<std::string, std::string> detect_keys_from_sample(const std::vector<std::string>& sample);
    std::string extract_value(const std::string& json, const std::string& key);
    bool fill_buffer();
};

class Tokenizer {
public:
    Tokenizer();
    void train(const std::vector<std::string>& texts, int num_merges);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    void save(const std::string& path) const;
    void load(const std::string& path);
    int vocab_size() const { return static_cast<int>(encoder_.size()); }

private:
    std::unordered_map<std::string, int> encoder_;
    std::unordered_map<int, std::string> decoder_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, std::vector<std::string>> cache_;

    std::vector<std::string> bpe(const std::string& token);
    std::vector<std::string> get_pairs(const std::vector<std::string>& word) const;
};

class UzaLEATCore {
public:
    UzaLEATCore(const CoreConfig& config);
    ~UzaLEATCore();
    int run();

private:
    CoreConfig config_;
    Tokenizer tokenizer_;
    GUTRProgram program_;
    ModelSO model_so_;

#ifdef UZALEAT_USE_VULKAN
    std::unique_ptr<uzagpt::VulkanBackend> vulkan_;
#endif

    static std::atomic<bool> interrupted_;
    static void signal_handler(int);

    bool load_model_so(const std::string& path);
    bool load_plugin();
    bool prepare_tokenizer();
    void train();
    void chat();
};

} // namespace

#endif
