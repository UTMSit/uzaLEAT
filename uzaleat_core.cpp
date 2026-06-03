#include "uzaleat_core.hpp"
#include "gutr_parser.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <climits>
#include <chrono>
#include <filesystem>
#include <csignal>
#include <unordered_set>
#include "vulkan_backend.hpp"

namespace uzaleat {

std::atomic<bool> UzaLEATCore::interrupted_{false};

namespace fs = std::filesystem;

// -------------------------------------------------------------------
// StreamingDataset implementation
// -------------------------------------------------------------------
StreamingDataset::StreamingDataset(const std::string& path, int shuffle_buffer)
    : shuffle_buffer_(shuffle_buffer), rng_(std::random_device{}()) {
    if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) files_.push_back(entry.path().string());
        }
    } else {
        files_.push_back(path);
    }
    if (files_.empty()) {
        std::cerr << "No files found in " << path << std::endl;
        exit(1);
    }
    current_file_idx_ = 0;
    open_next_file();
}

bool StreamingDataset::open_next_file() {
    if (current_file_.is_open()) current_file_.close();
    while (current_file_idx_ < files_.size()) {
        current_file_.open(files_[current_file_idx_]);
        if (current_file_.is_open()) {
            if (detect_keys_for_current_file()) {
                ++current_file_idx_;
                std::streampos pos = current_file_.tellg();
                std::string first;
                std::getline(current_file_, first);
                if (!first.empty() && first.front() == '[') in_array_ = true;
                current_file_.seekg(pos);
                return true;
            } else {
                std::cerr << "Warning: Could not detect keys in " << files_[current_file_idx_-1] << ", skipping.\n";
                current_file_.close();
            }
        }
        ++current_file_idx_;
    }
    return false;
}

bool StreamingDataset::detect_keys_for_current_file() {
    // Если ключи уже установлены явно (через set_keys или переменные окружения в конструкторе),
    // не перезаписывать их авто-детектом
    if (!current_input_key_.empty() && !current_output_key_.empty()) return true;

    std::vector<std::string> sample;
    std::string line;
    std::streampos start_pos = current_file_.tellg();
    for (int i = 0; i < 1000 && std::getline(current_file_, line); ++i) {
        if (line.empty() || line.front() == '[') continue;
        if (line.front() != '{') continue;
        sample.push_back(line);
    }
    current_file_.clear();
    current_file_.seekg(start_pos);
    if (sample.empty()) return false;
    std::tie(current_input_key_, current_output_key_) = detect_keys_from_sample(sample);
    return !current_input_key_.empty() && !current_output_key_.empty();
}

std::pair<std::string, std::string> StreamingDataset::detect_keys_from_sample(const std::vector<std::string>& sample) {
    static const std::vector<std::string> input_candidates = {
        "input", "prompt", "question", "instruction", "text", "content",
        "message", "query", "context", "src", "source", "user", "human"
    };
    static const std::vector<std::string> output_candidates = {
        "output", "completion", "answer", "response", "summary", "target",
        "label", "reply", "tgt", "assistant", "bot", "model"
    };
    std::unordered_map<std::string, int> input_counts, output_counts;
    for (const auto& line : sample) {
        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        for (const auto& key : input_candidates) {
            if (lower.find("\"" + key + "\"") != std::string::npos) input_counts[key]++;
        }
        for (const auto& key : output_candidates) {
            if (lower.find("\"" + key + "\"") != std::string::npos) output_counts[key]++;
        }
    }
    std::string best_input, best_output;
    int max_in = 0, max_out = 0;
    for (const auto& p : input_counts) if (p.second > max_in) { max_in = p.second; best_input = p.first; }
    for (const auto& p : output_counts) if (p.second > max_out) { max_out = p.second; best_output = p.first; }
    return {best_input, best_output};
}

std::string StreamingDataset::extract_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    while (pos < json.size() && json[pos] != ':') ++pos;
    if (pos >= json.size()) return "";
    ++pos;
    while (pos < json.size() && std::isspace(json[pos])) ++pos;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        size_t start = ++pos;
        while (pos < json.size()) {
            if (json[pos] == '"' && json[pos-1] != '\\') break;
            ++pos;
        }
        return json.substr(start, pos - start);
    } else if (json[pos] == '[' || json[pos] == '{') {
        size_t start = pos;
        int depth = 1;
        char open = json[pos];
        char close = (open == '{') ? '}' : ']';
        ++pos;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == open) ++depth;
            else if (json[pos] == close) --depth;
            ++pos;
        }
        return json.substr(start, pos - start);
    } else {
        size_t start = pos;
        while (pos < json.size() && !std::isspace(json[pos]) && json[pos] != ',' && json[pos] != '}') ++pos;
        return json.substr(start, pos - start);
    }
}

bool StreamingDataset::next(std::pair<std::string, std::string>& example) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (true) {
        if (shuffle_buffer_ > 0 && !shuffle_pool_.empty()) {
            example = shuffle_pool_.front();
            shuffle_pool_.pop();
            while (shuffle_pool_.size() < static_cast<size_t>(shuffle_buffer_) && fill_buffer()) {}
            return true;
        }
        std::string line;
        while (std::getline(current_file_, line)) {
            if (line.empty()) continue;
            if (in_array_) {
                if (line == "[") continue;
                if (line == "]") { in_array_ = false; continue; }
                if (line.back() == ',') line.pop_back();
            } else if (line.front() == '[') {
                in_array_ = true;
                continue;
            }
            if (line.front() != '{' || line.back() != '}') continue;
            std::string in = extract_value(line, current_input_key_);
            std::string out = extract_value(line, current_output_key_);
            if (in.empty()) {
                // fallback: попробовать "instruction" + "input"
                in = extract_value(line, "instruction");
                std::string inp = extract_value(line, "input");
                if (!inp.empty()) {
                    if (!in.empty()) in += " " + inp;
                    else in = inp;
                }
            }
            if (!in.empty() && !out.empty()) {
                example = {in, out};
                return true;
            }
        }
        if (!open_next_file()) return false;
    }
}

bool StreamingDataset::fill_buffer() {
    if (!current_file_.is_open() && !open_next_file()) return false;
    std::string line;
    while (shuffle_pool_.size() < static_cast<size_t>(shuffle_buffer_)) {
        if (!std::getline(current_file_, line)) {
            if (!open_next_file()) break;
            continue;
        }
        if (line.empty()) continue;
        if (in_array_) {
            if (line == "[") continue;
            if (line == "]") { in_array_ = false; continue; }
            if (line.back() == ',') line.pop_back();
        } else if (line.front() == '[') {
            in_array_ = true;
            continue;
        }
        if (line.front() != '{' || line.back() != '}') continue;
        std::string in = extract_value(line, current_input_key_);
        std::string out = extract_value(line, current_output_key_);
        if (in.empty()) {
            in = extract_value(line, "instruction");
            std::string inp = extract_value(line, "input");
            if (!inp.empty()) {
                if (!in.empty()) in += " " + inp;
                else in = inp;
            }
        }
        if (!in.empty() && !out.empty()) {
            shuffle_pool_.push({in, out});
        }
    }
    return !shuffle_pool_.empty();
}

void StreamingDataset::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_file_idx_ = 0;
    open_next_file();
    while (!shuffle_pool_.empty()) shuffle_pool_.pop();
    while (shuffle_pool_.size() < static_cast<size_t>(shuffle_buffer_) && fill_buffer()) {}
}

// -------------------------------------------------------------------
// Tokenizer — UTF-8 aware BPE
// -------------------------------------------------------------------
Tokenizer::Tokenizer() {}

std::vector<std::string> Tokenizer::get_pairs(const std::vector<std::string>& word) const {
    std::vector<std::string> pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i) pairs.push_back(word[i] + "|" + word[i+1]);
    return pairs;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) {
    auto it = cache_.find(token);
    if (it != cache_.end()) return it->second;
    std::vector<std::string> word;
    // Разбиваем на UTF-8 символы, а не байты
    for (size_t i = 0; i < token.size(); ) {
        unsigned char c = token[i];
        int len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        word.push_back(token.substr(i, len));
        i += len;
    }
    while (word.size() > 1) {
        std::string best_pair;
        int best_rank = INT_MAX;
        auto pairs = get_pairs(word);
        for (const auto& pair : pairs) {
            for (size_t i = 0; i < merges_.size(); ++i) {
                std::string merged = merges_[i].first + "|" + merges_[i].second;
                if (pair == merged && static_cast<int>(i) < best_rank) {
                    best_rank = static_cast<int>(i);
                    best_pair = pair;
                    break;
                }
            }
        }
        if (best_rank == INT_MAX) break;
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i+1 < word.size() && word[i] + "|" + word[i+1] == best_pair) {
                new_word.push_back(word[i] + word[i+1]);
                ++i;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = new_word;
    }
    cache_[token] = word;
    return word;
}

void Tokenizer::train(const std::vector<std::string>& texts, int num_merges) {
    // Собираем слова с UTF-8 aware разбивкой
    std::unordered_map<std::string, int> word_freq;
    for (const auto& text : texts) {
        std::string current_word;
        for (size_t i = 0; i < text.size(); ) {
            unsigned char c = text[i];
            int len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;

            std::string ch = text.substr(i, len);
            bool is_word_char = (len > 1) || std::isalnum(c) || c == '_';

            if (is_word_char) {
                current_word += ch;
            } else {
                if (!current_word.empty()) { word_freq[current_word]++; current_word.clear(); }
                word_freq[ch]++;
            }
            i += len;
        }
        if (!current_word.empty()) word_freq[current_word]++;
    }

    // Начальные символы — UTF-8 aware
    std::vector<std::pair<std::vector<std::string>, int>> word_symbols;
    for (const auto& [w, freq] : word_freq) {
        std::vector<std::string> symbols;
        for (size_t i = 0; i < w.size(); ) {
            unsigned char c = w[i];
            int len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            symbols.push_back(w.substr(i, len));
            i += len;
        }
        word_symbols.emplace_back(symbols, freq);
    }

    merges_.clear();
    for (int merge = 0; merge < num_merges; ++merge) {
        std::unordered_map<std::string, int> pair_freq;
        for (const auto& [syms, freq] : word_symbols) {
            if (syms.size() < 2) continue;
            for (size_t i = 0; i+1 < syms.size(); ++i)
                pair_freq[syms[i] + "|" + syms[i+1]] += freq;
        }
        if (pair_freq.empty()) break;
        std::string best_pair;
        int best_freq = 0;
        for (const auto& [p, f] : pair_freq) if (f > best_freq) { best_freq = f; best_pair = p; }
        size_t sep = best_pair.find('|');
        std::string first = best_pair.substr(0, sep);
        std::string second = best_pair.substr(sep+1);
        merges_.push_back({first, second});
        std::string merged = first + second;
        for (auto& [syms, freq] : word_symbols) {
            if (syms.size() < 2) continue;
            std::vector<std::string> new_syms;
            for (size_t i = 0; i < syms.size(); ++i) {
                if (i+1 < syms.size() && syms[i] == first && syms[i+1] == second) {
                    new_syms.push_back(merged);
                    ++i;
                } else new_syms.push_back(syms[i]);
            }
            syms = std::move(new_syms);
        }
    }

    encoder_.clear(); decoder_.clear();
    int idx = 0;
    // 256 байтовых токенов как fallback
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        encoder_[s] = idx;
        decoder_[idx] = s;
        ++idx;
    }
    // Добавляем все merge-результаты
    for (const auto& m : merges_) {
        std::string merged = m.first + m.second;
        if (encoder_.find(merged) == encoder_.end()) {
            encoder_[merged] = idx;
            decoder_[idx] = merged;
            ++idx;
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    // Разбиваем на UTF-8 символы
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = text[i];
        int len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        std::string ch = text.substr(i, len);
        auto it = encoder_.find(ch);
        if (it != encoder_.end()) {
            tokens.push_back(it->second);
        } else {
            // Fallback: побайтово
            for (int j = 0; j < len; ++j) {
                std::string byte(1, text[i+j]);
                auto it2 = encoder_.find(byte);
                if (it2 != encoder_.end()) tokens.push_back(it2->second);
            }
        }
        i += len;
    }
    // BPE слияние
    bool changed = true;
    while (changed && tokens.size() > 1) {
        changed = false;
        int best_rank = INT_MAX;
        size_t best_pos = 0;
        for (size_t i = 0; i+1 < tokens.size(); ++i) {
            std::string combined = decoder_[tokens[i]] + decoder_[tokens[i+1]];
            auto it = encoder_.find(combined);
            if (it != encoder_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_rank != INT_MAX) {
            tokens[best_pos] = best_rank;
            tokens.erase(tokens.begin() + best_pos + 1);
            changed = true;
        }
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;
    for (int t : tokens) {
        auto it = decoder_.find(t);
        if (it != decoder_.end()) result += it->second;
    }
    return result;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot save tokenizer to " + path);
    f << merges_.size() << "\n";
    for (const auto& m : merges_) f << m.first << " " << m.second << "\n";
}

void Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot load tokenizer from " + path);
    size_t n; f >> n;
    merges_.resize(n);
    for (size_t i = 0; i < n; ++i) f >> merges_[i].first >> merges_[i].second;
    encoder_.clear(); decoder_.clear();
    int idx = 0;
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        encoder_[s] = idx;
        decoder_[idx] = s;
        ++idx;
    }
    for (const auto& m : merges_) {
        std::string merged = m.first + m.second;
        if (encoder_.find(merged) == encoder_.end()) {
            encoder_[merged] = idx;
            decoder_[idx] = merged;
            ++idx;
        }
    }
}

// -------------------------------------------------------------------
// UzaLEATCore
// -------------------------------------------------------------------
void UzaLEATCore::signal_handler(int) { interrupted_ = true; }

UzaLEATCore::UzaLEATCore(const CoreConfig& config) : config_(config), model_so_{} {
    std::signal(SIGINT, signal_handler);
    #ifdef UZALEAT_USE_VULKAN
        if (config_.use_gpu) {
            uzagpt::GPUConfig gpu_cfg;
            gpu_cfg.staging_buffer_size_mb = 512;
            vulkan_ = std::make_unique<uzagpt::VulkanBackend>();
            if (!vulkan_->init(gpu_cfg)) {
                std::cerr << "Vulkan init failed, falling back to CPU\n";
                vulkan_.reset();
                config_.use_gpu = false;
            } else {
                g_vk_global = vulkan_.get();
            }
        }
    #endif
}

UzaLEATCore::~UzaLEATCore() {
    // Сначала выгружаем .so (статические данные плагина теряют доступ к GPU)
    if (model_so_.handle) {
        dlclose(model_so_.handle);
        model_so_.handle = nullptr;
    }
    // Потом выключаем Vulkan
    #ifdef UZALEAT_USE_VULKAN
        if (vulkan_) {
            vulkan_->shutdown();
            vulkan_.reset();
            g_vk_global = nullptr;
        }
    #endif
}

bool UzaLEATCore::load_model_so(const std::string& path) {
    model_so_.handle = dlopen(path.c_str(), RTLD_NOW);
    if (!model_so_.handle) {
        std::cerr << "Failed to load model .so: " << dlerror() << std::endl;
        return false;
    }
    model_so_.init       = (decltype(model_so_.init))       dlsym(model_so_.handle, "model_init");
    model_so_.train_step = (decltype(model_so_.train_step)) dlsym(model_so_.handle, "model_train_step");
    model_so_.generate   = (decltype(model_so_.generate))   dlsym(model_so_.handle, "model_generate");
    model_so_.save       = (decltype(model_so_.save))       dlsym(model_so_.handle, "model_save");
    model_so_.load       = (decltype(model_so_.load))       dlsym(model_so_.handle, "model_load");
    model_so_.get_tokenizer_size = (decltype(model_so_.get_tokenizer_size)) dlsym(model_so_.handle, "model_get_tokenizer_size");
    model_so_.get_tokenizer_data = (decltype(model_so_.get_tokenizer_data)) dlsym(model_so_.handle, "model_get_tokenizer_data");
    model_so_.load_holdout = (decltype(model_so_.load_holdout)) dlsym(model_so_.handle, "model_load_holdout");

    if (!model_so_.init || !model_so_.train_step || !model_so_.generate || !model_so_.save || !model_so_.load) {
        std::cerr << "Model .so missing required symbols (need: model_init, model_train_step, model_generate, model_save, model_load)" << std::endl;
        dlclose(model_so_.handle);
        model_so_.handle = nullptr;
        return false;
    }
    std::cout << "Model .so loaded: " << path << std::endl;
    return true;
}

bool UzaLEATCore::load_plugin() {
    std::ifstream file(config_.plugin_path);
    if (!file) {
        std::cerr << "Cannot open plugin: " << config_.plugin_path << std::endl;
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    GUTRParser parser;
    return parser.parse(content, program_);
}

bool UzaLEATCore::prepare_tokenizer() {
    if (!config_.tokenizer_path.empty() && std::ifstream(config_.tokenizer_path).good()) {
        try {
            tokenizer_.load(config_.tokenizer_path);
            std::cout << "Loaded tokenizer, vocab size = " << tokenizer_.vocab_size() << std::endl;
            config_.vocab_size = tokenizer_.vocab_size();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load tokenizer: " << e.what() << ", will train from data\n";
        }
    }
    // Принудительно установить ключи из переменных окружения
    StreamingDataset sample_ds(config_.data_path, 0);
    const char* env_in = std::getenv("UZALEAT_INPUT_KEY");
    const char* env_out = std::getenv("UZALEAT_OUTPUT_KEY");
    if (env_in && env_out) {
        sample_ds.current_input_key_ = env_in;
        sample_ds.current_output_key_ = env_out;
        sample_ds.reset();
    }
    std::vector<std::string> texts;
    std::pair<std::string, std::string> ex;
    int count = 0;
    while (sample_ds.next(ex) && count < 100000) {
        texts.push_back(ex.first);
        texts.push_back(ex.second);
        ++count;
    }
    if (texts.empty()) {
        std::cerr << "No data for tokenizer training\n";
        return false;
    }
    tokenizer_.train(texts, 5000);
    std::string tokenizer_save_path = config_.tokenizer_path.empty() ? "tokenizer.bin" : config_.tokenizer_path;
    tokenizer_.save(tokenizer_save_path);
    std::cout << "Trained tokenizer, vocab size = " << tokenizer_.vocab_size() << std::endl;
    config_.vocab_size = tokenizer_.vocab_size();
    return true;
}

void UzaLEATCore::train() {
    if (!prepare_tokenizer()) return;

    bool use_so = !config_.model_so_path.empty();
    bool use_gutr = !config_.plugin_path.empty();

    if (use_so) {
        if (!load_model_so(config_.model_so_path)) return;
    }

    if (!config_.model_path.empty() && std::ifstream(config_.model_path).good()) {
        std::cout << "Loading existing model from " << config_.model_path << std::endl;
        if (use_so) model_so_.load(config_.model_path.c_str());
        else program_.load(config_.model_path);
    } else {
        if (use_so) {
            model_so_.init(config_.hidden_size, config_.num_layers, config_.vocab_size,
                           config_.context_size, config_.tt_rank, config_.num_experts,
                           config_.window_size, config_.update_interval);
        } if (use_so && model_so_.load_holdout) {
            std::ifstream hf("holdout.jsonl");
            if (hf.good()) {
                hf.close();
                model_so_.load_holdout("holdout.jsonl");
            }
        } else if (use_gutr) {
        if (!load_plugin()) return;
        program_.set_global("H", GUTRValue::integer(config_.hidden_size));
        program_.set_global("L", GUTRValue::integer(config_.num_layers));
        program_.set_global("V", GUTRValue::integer(config_.vocab_size));
        program_.set_global("ctx_len", GUTRValue::integer(config_.context_size));
        program_.set_global("tt_rank", GUTRValue::integer(config_.tt_rank));
        program_.set_global("proj_rank", GUTRValue::integer(config_.proj_rank));
        program_.set_global("update_interval", GUTRValue::integer(config_.update_interval));
        program_.set_global("num_experts", GUTRValue::integer(config_.num_experts));
        program_.set_global("window_size", GUTRValue::integer(config_.window_size));
        program_.init("{}");
    } else {
        std::cerr << "Error: need --plugin or --model-so\n";
        return;
        }
    }

    StreamingDataset dataset(config_.data_path, config_.shuffle_buffer);
    size_t step = 0;
    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        dataset.reset();
        int steps = 0;
        std::pair<std::string, std::string> ex;
        while (dataset.next(ex)) {
            if (interrupted_) {
                std::string save_path = config_.model_path.empty() ? "interrupted.gguf" : config_.model_path;
                std::cout << "\nInterrupted. Saving to " << save_path << "...\n";
                if (use_so) model_so_.save(save_path.c_str());
                else program_.save(save_path);
                return;
            }
            auto inp = tokenizer_.encode(ex.first);
            auto tgt = tokenizer_.encode(ex.second);
            if (inp.empty() || tgt.empty()) continue;
            size_t seq_len = std::min({inp.size(), tgt.size(), config_.context_size});
            inp.resize(seq_len);
            tgt.resize(seq_len);

            if (use_so)
                model_so_.train_step(inp.data(), tgt.data(), seq_len, config_.learning_rate);
            else
                program_.train_step(inp, tgt, config_.learning_rate);

            ++steps; ++step;
            if (steps % 10 == 0)
                std::cout << "\rEpoch " << epoch+1 << " Step " << steps << std::flush;

            if (step % config_.sample_every == 0) {
                std::vector<int> gen(config_.max_tokens_sample);
                int len;
                if (use_so)
                    len = model_so_.generate(inp.data(), inp.size(), gen.data(), config_.max_tokens_sample,
                                            config_.temperature, config_.top_p);
                else
                    len = program_.generate(inp, gen, config_.max_tokens_sample, config_.temperature, config_.top_p);
                std::cout << "\n[Sample] " << tokenizer_.decode(std::vector<int>(gen.begin(), gen.begin()+len)) << "\n";
            }
        }
        std::cout << "\nEpoch " << epoch+1 << " done.\n";
        std::string ckpt = "checkpoint_epoch_" + std::to_string(epoch+1) + ".gguf";
        if (use_so) model_so_.save(ckpt.c_str());
        else program_.save(ckpt);
    }
    std::string final_path = config_.model_path.empty() ? "final.gguf" : config_.model_path;
    if (use_so) model_so_.save(final_path.c_str());
    else program_.save(final_path);
    std::cout << "Training complete. Model saved to " << final_path << std::endl;

    // Очистить GPU-буферы плагина
    if (use_so && model_so_.handle) {
        auto cleanup = (void (*)()) dlsym(model_so_.handle, "model_cleanup");
        if (cleanup) cleanup();
    }
}

void UzaLEATCore::chat() {
    bool use_so = !config_.model_so_path.empty();
    bool use_gutr = !config_.plugin_path.empty();

    if (use_so) {
        if (!load_model_so(config_.model_so_path)) return;
        // Загружаем модель (токенизатор загрузится внутри model_load, если он там есть)
        model_so_.load(config_.model_path.c_str());
    } else if (use_gutr) {
        if (!load_plugin()) return;
        program_.load(config_.model_path);
    } else {
        std::cerr << "Error: need --plugin or --model-so\n";
        return;
    }

    // Пробуем загрузить токенизатор из .so модели, если он там встроен
    if (use_so && model_so_.get_tokenizer_size && model_so_.get_tokenizer_data) {
        size_t sz = model_so_.get_tokenizer_size();
        if (sz > 0) {
            std::vector<char> buf(sz);
            model_so_.get_tokenizer_data(buf.data(), sz);
            // Сохраняем во временный файл и загружаем через существующий tokenizer_.load()
            std::string tmp_path = "_tok_cache.bin";
            std::ofstream tmp(tmp_path, std::ios::binary);
            tmp.write(buf.data(), sz);
            tmp.close();
            tokenizer_.load(tmp_path);
            std::remove(tmp_path.c_str());
            std::cout << "Loaded tokenizer from model (vocab=" << tokenizer_.vocab_size() << ")" << std::endl;
        }
    }
    // Пробуем загрузить токенизатор из отдельного файла
    if (!config_.tokenizer_path.empty() && std::ifstream(config_.tokenizer_path).good()) {
        try { tokenizer_.load(config_.tokenizer_path); }
        catch (const std::exception& e) {
            std::cerr << "Failed to load tokenizer from file: " << e.what() << std::endl;
        }
    }

    // Если токенизатор не загрузился ни из файла, ни из GGUF — ошибка
    if (tokenizer_.vocab_size() == 0) {
        std::cerr << "Tokenizer not found. Provide --tokenizer or use a GGUF with embedded tokenizer.\n";
        return;
    }

    std::cout << "Tokenizer loaded, vocab size = " << tokenizer_.vocab_size() << std::endl;
    std::cout << "Chat mode. Type 'quit' to exit.\n";

    std::string prompt;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, prompt);
        if (prompt == "quit") break;
        auto inp = tokenizer_.encode(prompt);
        std::vector<int> gen(config_.max_tokens_sample);
        int len;
        if (use_so)
            len = model_so_.generate(inp.data(), inp.size(), gen.data(), config_.max_tokens_sample,
                                    config_.temperature, config_.top_p);
        else
            len = program_.generate(inp, gen, config_.max_tokens_sample, config_.temperature, config_.top_p);
        std::cout << tokenizer_.decode(std::vector<int>(gen.begin(), gen.begin()+len)) << std::endl;
    }
}

int UzaLEATCore::run() {
    if (config_.train_mode) train();
    else if (config_.chat_mode) chat();
    else return 1;
    return 0;
}

} // namespace
