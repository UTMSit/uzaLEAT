// файл: main.cpp
#include <iostream>
#include <csignal>
#include <getopt.h>
#include <omp.h>
#include "uzaleat_core.hpp"

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --plugin FILE           .gutr plugin file (required)\n"
              << "  --train                 Training mode\n"
              << "  --chat                  Chat mode\n"
              << "  --data PATH             Training data path (file or directory with JSON lines)\n"
              << "  --tokenizer FILE        Tokenizer file (load/save)\n"
              << "  --model FILE            Model file (output/input GGUF)\n"
              << "  --hidden N              Hidden size (default 768)\n"
              << "  --layers N              Number of layers (default 12)\n"
              << "  --context N             Context size (default 1024)\n"
              << "  --epochs N              Epochs (default 3)\n"
              << "  --lr F                  Learning rate (default 0.0001)\n"
              << "  --batch N               Batch size (default 8)\n"
              << "  --threads N             CPU threads (default 4)\n"
              << "  --gpu                   Enable Vulkan GPU acceleration\n"
              << "  --sample-every N        Sample during training every N steps (default 50)\n"
              << "  --temperature F         Sampling temperature (default 0.8)\n"
              << "  --top-p F               Top-p sampling (default 0.9)\n"
              << "  --max-tokens N          Max tokens to generate (default 50)\n"
              << "  --shuffle-buffer N      Shuffle buffer size (default 10000)\n"
              << "  --help                  Show this help\n";
    exit(0);
}

int main(int argc, char* argv[]) {
    uzaleat::CoreConfig config;

    static struct option long_opts[] = {
        {"plugin",         required_argument, 0, 0},
        {"train",          no_argument,       0, 0},
        {"chat",           no_argument,       0, 0},
        {"data",           required_argument, 0, 0},
        {"tokenizer",      required_argument, 0, 0},
        {"model",          required_argument, 0, 0},
        {"hidden",         required_argument, 0, 0},
        {"layers",         required_argument, 0, 0},
        {"context",        required_argument, 0, 0},
        {"epochs",         required_argument, 0, 0},
        {"lr",             required_argument, 0, 0},
        {"batch",          required_argument, 0, 0},
        {"threads",        required_argument, 0, 0},
        {"gpu",            no_argument,       0, 0},
        {"sample-every",   required_argument, 0, 0},
        {"temperature",    required_argument, 0, 0},
        {"top-p",          required_argument, 0, 0},
        {"max-tokens",     required_argument, 0, 0},
        {"shuffle-buffer", required_argument, 0, 0},
        {"help",           no_argument,       0, 'h'},
        {0,0,0,0}
    };

    int c, option_index = 0;
    while ((c = getopt_long(argc, argv, "h", long_opts, &option_index)) != -1) {
        if (c == 'h') print_usage(argv[0]);
        if (c != 0) continue;

        const std::string opt = long_opts[option_index].name;
        if (opt == "plugin")         config.plugin_path = optarg;
        else if (opt == "train")     config.train_mode = true;
        else if (opt == "chat")      config.chat_mode = true;
        else if (opt == "data")      config.data_path = optarg;
        else if (opt == "tokenizer") config.tokenizer_path = optarg;
        else if (opt == "model")     config.model_path = optarg;
        else if (opt == "hidden")    config.hidden_size = std::stoul(optarg);
        else if (opt == "layers")    config.num_layers = std::stoul(optarg);
        else if (opt == "context")   config.context_size = std::stoul(optarg);
        else if (opt == "epochs")    config.epochs = std::stoi(optarg);
        else if (opt == "lr")        config.learning_rate = std::stof(optarg);
        else if (opt == "batch")     config.batch_size = std::stoi(optarg);
        else if (opt == "threads")   config.num_threads = std::stoi(optarg);
        else if (opt == "gpu")       config.use_gpu = true;
        else if (opt == "sample-every")   config.sample_every = std::stoi(optarg);
        else if (opt == "temperature")    config.temperature = std::stof(optarg);
        else if (opt == "top-p")          config.top_p = std::stof(optarg);
        else if (opt == "max-tokens")     config.max_tokens_sample = std::stoi(optarg);
        else if (opt == "shuffle-buffer") config.shuffle_buffer = std::stoi(optarg);
        else if (opt == "help")      print_usage(argv[0]);
    }

    if (!config.train_mode && !config.chat_mode) {
        std::cerr << "Error: specify --train or --chat\n";
        print_usage(argv[0]);
    }
    if (config.plugin_path.empty()) {
        std::cerr << "Error: --plugin required\n";
        print_usage(argv[0]);
    }

#ifdef _OPENMP
    omp_set_num_threads(config.num_threads);
#endif

    uzaleat::UzaLEATCore core(config);
    return core.run();
}
