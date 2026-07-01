// ============================================================================
// flettohm_lite_transformers.cpp — FLETTOHM Lite for Transformers
// Public Release v1.0 | MIT Licenses
// ============================================================================

#include "vulkan_backend.hpp"
#include <random>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>

extern uzagpt::VulkanBackend* g_vk_global;  // определён в vulkan_backend.cpp (линкуется в uzaLEAT)
// В этом плагине доступ к бэкенду идёт через ссылочный алиас на g_vk_global,
// чтобы корректно резолвиться при dlopen из основного бинаря.
static uzagpt::VulkanBackend*& g_vk = g_vk_global;

// ============================================================================
// КОНФИГУРАЦИЯ
// ============================================================================
static int H = 768;           // hidden size
static int L = 12;            // layers
static int V = 32000;         // vocab
static int CTX = 8192;        // context
static int HEAD_DIM = 64;
static int NUM_HEADS = H / HEAD_DIM;
static int RANK = 64;          // rank for random projection
static float BLEND = 0.1f;
static float LAMBDA = 0.001f;
static int UPDATE_EVERY = 1000;
static float TEMP = 0.8f;
static float TOP_P = 0.9f;

// ============================================================================
// ГЛОБАЛЬНЫЕ БУФЕРЫ
// ============================================================================
struct GPULayer {
    // Attention
    uzagpt::GPUBuffer* q_proj;
    uzagpt::GPUBuffer* k_proj;
    uzagpt::GPUBuffer* v_proj;
    uzagpt::GPUBuffer* o_proj;

    // FFN (SwiGLU)
    uzagpt::GPUBuffer* gate_proj;
    uzagpt::GPUBuffer* up_proj;
    uzagpt::GPUBuffer* down_proj;

    // RMS Norm
    uzagpt::GPUBuffer* rms_attn;
    uzagpt::GPUBuffer* rms_ffn;

    // FLETTOHM статистики
    uzagpt::GPUBuffer* XTX;
    uzagpt::GPUBuffer* XTY;
    uzagpt::GPUBuffer* R_proj;
    uzagpt::GPUBuffer* temp_proj;

    int token_count;
    size_t size_q, size_k, size_v, size_o;
    size_t size_gate, size_up, size_down;
    uzagpt::GPUBuffer* W_small;
    uzagpt::GPUBuffer* grad;
    uzagpt::GPUBuffer* W_new;
};

static std::vector<GPULayer> g_layers;
static uzagpt::GPUBuffer* wte;
static uzagpt::GPUBuffer* lm_head;
static uzagpt::GPUBuffer* rms_final;
static uzagpt::GPUBuffer* lm_head_XTX;
static uzagpt::GPUBuffer* lm_head_XTY;
static uzagpt::GPUBuffer* lm_head_R_proj;
static uzagpt::GPUBuffer* lm_head_temp_proj;
static uzagpt::GPUBuffer* lm_head_W_small;
static uzagpt::GPUBuffer* lm_head_grad;
static uzagpt::GPUBuffer* lm_head_W_new;
static std::vector<uzagpt::GPUBuffer*> kv_cache_k;
static std::vector<uzagpt::GPUBuffer*> kv_cache_v;
static std::vector<float> cpu_buffer;  // для редких выгрузок

// ============================================================================
// БАЗОВЫЕ GPU-ОПЕРАЦИИ (обёртки над vk_global)
// ============================================================================
static void matmul(uzagpt::GPUBuffer* A, uzagpt::GPUBuffer* B, uzagpt::GPUBuffer* C,
                   uint32_t M, uint32_t K, uint32_t N, bool transA, bool transB) {
    g_vk->matmul(A, B, C, M, K, N, transA, transB);
}

static void rms_norm(uzagpt::GPUBuffer* x, uzagpt::GPUBuffer* weight, uzagpt::GPUBuffer* out, uint32_t n) {
    g_vk->rms_norm(x, weight, out, n);
}

static void add(uzagpt::GPUBuffer* a, uzagpt::GPUBuffer* b, uzagpt::GPUBuffer* out, uint32_t n) {
    g_vk->element_wise_add(a, b, out, n);
}

static void mul(uzagpt::GPUBuffer* a, uzagpt::GPUBuffer* b, uzagpt::GPUBuffer* out, uint32_t n) {
    g_vk->element_wise_mul(a, b, out, n);
}

static void gelu(uzagpt::GPUBuffer* in, uzagpt::GPUBuffer* out, uint32_t n) {
    g_vk->gelu(in, out, n);
}

static void softmax(uzagpt::GPUBuffer* in, uzagpt::GPUBuffer* out, uint32_t rows, uint32_t cols) {
    g_vk->softmax(in, out, rows, cols);
}

static void upload(uzagpt::GPUBuffer* dst, const void* src, size_t size) {
    g_vk->upload_to_buffer(dst, src, size);
}

static void download(void* dst, uzagpt::GPUBuffer* src, size_t size) {
    g_vk->download_from_buffer(dst, src, size);
}

// ============================================================================
// SWIGLU (GPU шейдер)
// ============================================================================
static void swiglu(uzagpt::GPUBuffer* gate, uzagpt::GPUBuffer* up, uzagpt::GPUBuffer* out, uint32_t n) {
    // out = gate * sigmoid(gate) * up
    // Запускаем compute shader
    g_vk->swiglu(gate, up, out, n);
}

// ============================================================================
// ATTENTION С KV КЭШЕМ (полный GPU шейдер)
// ============================================================================
static void attention_with_cache(uzagpt::GPUBuffer* q,           // [NUM_HEADS, HEAD_DIM]
                                 uzagpt::GPUBuffer* k_cache,     // [CTX, NUM_HEADS, HEAD_DIM]
                                 uzagpt::GPUBuffer* v_cache,     // [CTX, NUM_HEADS, HEAD_DIM]
                                 int pos,                        // текущая позиция
                                 int seq_len,                    // длина последовательности в кэше
                                 uzagpt::GPUBuffer* out) {       // [H]
    g_vk->attention(q, k_cache, v_cache, out, pos, seq_len, NUM_HEADS, HEAD_DIM);
}

// ============================================================================
// FUSED QKV (один шейдер, три матрицы)
// ============================================================================
static void fused_qkv(uzagpt::GPUBuffer* x,
                      uzagpt::GPUBuffer* q_proj,
                      uzagpt::GPUBuffer* k_proj,
                      uzagpt::GPUBuffer* v_proj,
                      uzagpt::GPUBuffer* q,
                      uzagpt::GPUBuffer* k,
                      uzagpt::GPUBuffer* v,
                      uint32_t d_model) {
    g_vk->fused_qkv(x, q_proj, k_proj, v_proj, q, k, v, d_model);
}

// ============================================================================
// ДОБАВЛЕНИЕ РЕГУЛЯРИЗАЦИИ К ДИАГОНАЛИ (GPU)
// ============================================================================
static void add_diagonal(uzagpt::GPUBuffer* mat, uint32_t size, float lambda) {
    g_vk->add_diagonal(mat, size, lambda);
}

// ============================================================================
// ПЛАВНОЕ ОБНОВЛЕНИЕ ВЕСОВ (GPU)
// ============================================================================
static void blend_weights(uzagpt::GPUBuffer* W, uzagpt::GPUBuffer* W_new, uint32_t n, float blend) {
    g_vk->blend_weights(W, W_new, n, blend);
}

// ============================================================================
// FLETTOHM: обновление W через решение (XTX + λI) W_small = XTY на GPU
//
//   W ≈ R^T @ W_small,   dim(W) = [in_dim, out_dim],  R: [RANK, in_dim]
//   W_small: [RANK, out_dim]
//   XTX:    [RANK, RANK]   (накапливается во flettohm_accumulate)
//   XTY:    [RANK, out_dim]
//
//   Решаем 5 итерациями градиентного спуска на W_small (RANK x RANK крошечная),
//   проецируем обратно в W и блендим. W_small/grad/W_new берутся из слоя —
//   размер выделяется под максимум по всем матрицам слоя (out_max=4H, in_max=4H).
// ============================================================================
static void flettohm_update(uzagpt::GPUBuffer* W, GPULayer& layer,
                            uint32_t out_dim, uint32_t in_dim) {
    if (layer.token_count < UPDATE_EVERY) return;

    const uint32_t out_max = 4 * (uint32_t)H;
    const uint32_t in_max  = 4 * (uint32_t)H;
    if (out_dim > out_max || in_dim > in_max) {
        // Защита: общие буферы слоя выделены под максимум 4H x 4H
        return;
    }

    // 1. XTX += LAMBDA * I
    g_vk->add_diagonal(layer.XTX, RANK, LAMBDA);

    // 2. Градиентный спуск по W_small
    const float lr = 0.1f;
    const uint32_t wsmall_n = (uint32_t)RANK * out_dim;
    for (int iter = 0; iter < 5; iter++) {
        g_vk->matmul(layer.XTX, layer.W_small, layer.grad,
                     RANK, RANK, out_dim, false, false);
        g_vk->element_wise_sub(layer.grad, layer.XTY, layer.grad, wsmall_n);
        g_vk->element_wise_mul_scalar(layer.grad, lr, layer.grad);
        g_vk->element_wise_sub(layer.W_small, layer.grad, layer.W_small, wsmall_n);
    }

    // 3. W_new = R^T @ W_small
    g_vk->matmul(layer.R_proj, layer.W_small, layer.W_new,
                 in_dim, RANK, out_dim, true, false);

    // 4. Blend
    g_vk->blend_weights(W, layer.W_new, in_dim * out_dim, BLEND);

    // 5. Сброс статистик
    g_vk->zero_buffer(layer.XTX);
    g_vk->zero_buffer(layer.XTY);
    layer.token_count = 0;
}

// ============================================================================
// ВОССТАНОВЛЕНИЕ W = R^T @ W_small (GPU)
// ============================================================================
static void reconstruct_weight(uzagpt::GPUBuffer* R,        // [rank, d]
                               uzagpt::GPUBuffer* W_small,  // [rank, out]
                               uzagpt::GPUBuffer* W,        // [d, out]
                               uint32_t d, uint32_t out_dim, uint32_t rank) {
    g_vk->matmul(R, W_small, W, d, rank, out_dim, true, false);
}

// ============================================================================
// ZERO BUFFER
// ============================================================================
static void zero_buffer(uzagpt::GPUBuffer* buf) {
    g_vk->zero_buffer(buf);
}

// ============================================================================
// СЭМПЛИНГ НА GPU
// ============================================================================
static int sample_gpu(uzagpt::GPUBuffer* logits, int vocab_size, float temp, float top_p) {
    if (temp <= 0.01f) {
        // Argmax — берём самый вероятный токен
        return g_vk->argmax(logits, vocab_size);
    }

    // Скачиваем на CPU для сэмплинга (только это вынужденно на CPU)
    if (cpu_buffer.size() < (size_t)vocab_size) {
        cpu_buffer.resize(vocab_size);
    }
    download(cpu_buffer.data(), logits, vocab_size * sizeof(float));

    // Softmax + temperature
    float mx = *std::max_element(cpu_buffer.begin(), cpu_buffer.begin() + vocab_size);
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) {
        cpu_buffer[i] = std::exp((cpu_buffer[i] - mx) / temp);
        sum += cpu_buffer[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        cpu_buffer[i] *= inv_sum;
    }

    // Top-p sampling
    if (top_p < 0.99f) {
        std::vector<std::pair<float, int>> probs(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = {cpu_buffer[i], i};
        }
        std::sort(probs.begin(), probs.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        float cumsum = 0;
        int cutoff = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i].first;
            cutoff = i + 1;
            if (cumsum >= top_p) break;
        }

        float r = (float)rand() / RAND_MAX;
        cumsum = 0;
        for (int i = 0; i < cutoff; i++) {
            cumsum += probs[i].first;
            if (r < cumsum) return probs[i].second;
        }
        return probs[0].second;
    }

    // Генерация случайного токена
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += cpu_buffer[i];
        if (r < cumsum) return i;
    }
    return vocab_size - 1;
}

// ============================================================================
// FLETTOHM НАКОПЛЕНИЕ (GPU)
//   x: [in_dim]   вход
//   y: [out_dim]  целевая активация
//   temp_proj: [RANK]
//   XTX: [RANK, RANK]   += (R@x)(R@x)^T
//   XTY: [RANK, out_dim] += (R@x) y^T
// ============================================================================
static void flettohm_accumulate(uzagpt::GPUBuffer* R_proj, uzagpt::GPUBuffer* temp_proj,
                                uzagpt::GPUBuffer* XTX, uzagpt::GPUBuffer* XTY,
                                uzagpt::GPUBuffer* x, uzagpt::GPUBuffer* y,
                                uint32_t in_dim, uint32_t out_dim) {
    // x_proj = R @ x   : [RANK]
    matmul(R_proj, x, temp_proj, RANK, in_dim, 1, false, false);

    // XTX += x_proj @ x_proj^T  : [RANK, RANK]
    matmul(temp_proj, temp_proj, XTX, RANK, 1, RANK, false, true);

    // XTY += x_proj @ y^T       : [RANK, out_dim]
    matmul(temp_proj, y, XTY, RANK, 1, out_dim, false, true);
}

// ============================================================================
// ОБНОВЛЕНИЕ ВЕСОВ ЧЕРЕЗ FLETTOHM (полностью на GPU)
// ============================================================================
static void flettohm_update(uzagpt::GPUBuffer* W, GPULayer& layer,
                            uint32_t out_dim, uint32_t in_dim,
                            uzagpt::GPUBuffer* W_small, uzagpt::GPUBuffer* grad, uzagpt::GPUBuffer* W_new) {
    if (layer.token_count < UPDATE_EVERY) return;

    // Регуляризация
    g_vk->add_diagonal(layer.XTX, RANK, LAMBDA);

    // 5 итераций градиентного спуска
    float lr = 0.1f;
    for (int iter = 0; iter < 5; iter++) {
        // grad = XTX @ W_small
        g_vk->matmul(layer.XTX, W_small, grad, RANK, RANK, out_dim, false, false);
        // grad = grad - XTY
        g_vk->element_wise_sub(grad, layer.XTY, grad, RANK * out_dim);
        // W_small = W_small - lr * grad
        g_vk->element_wise_mul_scalar(grad, lr, grad);
        g_vk->element_wise_sub(W_small, grad, W_small, RANK * out_dim);
    }

    // W_new = R^T @ W_small
    g_vk->matmul(layer.R_proj, W_small, W_new, in_dim, RANK, out_dim, true, false);

    // Blend
    g_vk->blend_weights(W, W_new, in_dim * out_dim, BLEND);

    // Сброс
    g_vk->zero_buffer(layer.XTX);
    g_vk->zero_buffer(layer.XTY);
    layer.token_count = 0;
}

// ============================================================================
// FORWARD PASS (чистый GPU, все операции через шейдеры)
// ============================================================================
static void forward_token(int token_id, int pos,
                          std::vector<uzagpt::GPUBuffer*>& k_caches,
                          std::vector<uzagpt::GPUBuffer*>& v_caches,
                          uzagpt::GPUBuffer* logits_out,
                          bool collect_stats) {

    // 1. Эмбеддинг токена (row из wte)
    static uzagpt::GPUBuffer* x = nullptr;
    static uzagpt::GPUBuffer* residual = nullptr;
    if (!x) {
        x = g_vk->allocate_buffer(H * sizeof(float));
        residual = g_vk->allocate_buffer(H * sizeof(float));
    }

    // Извлекаем строку token_id из матрицы wte
    g_vk->extract_row(wte, token_id, x, H);

    // 2. Проход по слоям
    for (int l = 0; l < L; l++) {
        auto& ly = g_layers[l];

        // RMS Norm 1 + residual
        rms_norm(x, ly.rms_attn, residual, H);

        // Fused QKV
        static uzagpt::GPUBuffer* q = nullptr;
        static uzagpt::GPUBuffer* k = nullptr;
        static uzagpt::GPUBuffer* v = nullptr;
        if (!q) {
            q = g_vk->allocate_buffer(H * sizeof(float));
            k = g_vk->allocate_buffer(H * sizeof(float));
            v = g_vk->allocate_buffer(H * sizeof(float));
        }
        fused_qkv(residual, ly.q_proj, ly.k_proj, ly.v_proj, q, k, v, H);

        // Attention with KV cache
        static uzagpt::GPUBuffer* attn_out = nullptr;
        if (!attn_out) attn_out = g_vk->allocate_buffer(H * sizeof(float));
        attention_with_cache(q, k_caches[l], v_caches[l], pos, pos + 1, attn_out);

        // Residual 1
        add(x, attn_out, x, H);

        // RMS Norm 2
        rms_norm(x, ly.rms_ffn, residual, H);

        // SwiGLU FFN
        static uzagpt::GPUBuffer* gate = nullptr;
        static uzagpt::GPUBuffer* up = nullptr;
        static uzagpt::GPUBuffer* ffn = nullptr;
        static uzagpt::GPUBuffer* ffn_out = nullptr;
        if (!gate) {
            gate = g_vk->allocate_buffer(4*H * sizeof(float));
            up = g_vk->allocate_buffer(4*H * sizeof(float));
            ffn = g_vk->allocate_buffer(4*H * sizeof(float));
            ffn_out = g_vk->allocate_buffer(H * sizeof(float));
        }

        matmul(ly.gate_proj, residual, gate, 4*H, H, 1, false, false);
        matmul(ly.up_proj, residual, up, 4*H, H, 1, false, false);
        swiglu(gate, up, ffn, 4*H);
        matmul(ly.down_proj, ffn, ffn_out, H, 4*H, 1, false, false);

        // Residual 2
        add(x, ffn_out, x, H);

        // Сбор статистик для FLETTOHM (если нужно)
        if (collect_stats) {
            // x = вход residual-блока, y = выход residual-блока (оба [H])
            // Накапливаем XTX += (R@x)(R@x)^T, XTY += (R@x) y^T
            flettohm_accumulate(ly.R_proj, ly.temp_proj,
                                ly.XTX, ly.XTY,
                                residual, x,
                                (uint32_t)H, (uint32_t)H);
            ly.token_count++;
        }
    }

    // 3. Final RMS Norm + LM Head
    rms_norm(x, rms_final, residual, H);
    matmul(lm_head, residual, logits_out, V, H, 1, false, false);
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
// ============================================================================
extern "C" void model_init(int hidden, int layers_cnt, int vocab, int ctx,
                           int rank, int update_every) {
    H = hidden;
    L = layers_cnt;
    V = vocab;
    CTX = ctx;
    RANK = rank;
    UPDATE_EVERY = update_every;
    NUM_HEADS = H / HEAD_DIM;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 0.02);

    // WTE
    wte = g_vk->allocate_buffer(V * H * sizeof(float));
    std::vector<float> wte_data(V * H);
    for (auto& v : wte_data) v = dist(rng);
    upload(wte, wte_data.data(), V * H * sizeof(float));

    // LM Head
    lm_head = g_vk->allocate_buffer(V * H * sizeof(float));
    upload(lm_head, wte_data.data(), V * H * sizeof(float));

    // RMS Final
    rms_final = g_vk->allocate_buffer(H * sizeof(float));
    std::vector<float> rms_data(H, 1.0f);
    upload(rms_final, rms_data.data(), H * sizeof(float));

    // KV Cache
    kv_cache_k.resize(L);
    kv_cache_v.resize(L);
    size_t cache_per_layer = CTX * NUM_HEADS * HEAD_DIM * sizeof(float);
    for (int l = 0; l < L; l++) {
        kv_cache_k[l] = g_vk->allocate_buffer(cache_per_layer);
        kv_cache_v[l] = g_vk->allocate_buffer(cache_per_layer);
        zero_buffer(kv_cache_k[l]);
        zero_buffer(kv_cache_v[l]);
    }

    // Слои
    g_layers.resize(L);
    for (int l = 0; l < L; l++) {
        auto& ly = g_layers[l];

        ly.q_proj = g_vk->allocate_buffer(H * H * sizeof(float));
        ly.k_proj = g_vk->allocate_buffer(H * H * sizeof(float));
        ly.v_proj = g_vk->allocate_buffer(H * H * sizeof(float));
        ly.o_proj = g_vk->allocate_buffer(H * H * sizeof(float));
        ly.gate_proj = g_vk->allocate_buffer(H * 4*H * sizeof(float));
        ly.up_proj = g_vk->allocate_buffer(H * 4*H * sizeof(float));
        ly.down_proj = g_vk->allocate_buffer(4*H * H * sizeof(float));
        ly.rms_attn = g_vk->allocate_buffer(H * sizeof(float));
        ly.rms_ffn = g_vk->allocate_buffer(H * sizeof(float));
        ly.XTX = g_vk->allocate_buffer(RANK * RANK * sizeof(float));
        // XTY: [RANK, max_out_dim]. max_out_dim = 4*H (gate/up/down=4H, q/k/v/o=H, lm_head=V)
        const uint32_t max_out_dim = 4 * (uint32_t)H;
        const uint32_t max_in_dim  = 4 * (uint32_t)H;
        ly.XTY = g_vk->allocate_buffer((size_t)RANK * max_out_dim * sizeof(float));
        ly.R_proj = g_vk->allocate_buffer((size_t)RANK * max_in_dim * sizeof(float));
        ly.temp_proj = g_vk->allocate_buffer(RANK * sizeof(float));

        // W_small/grad/W_new — общие для всех матриц слоя (обновляются последовательно),
        // размер под максимум: out_max=4H, in_max=4H
        ly.W_small = g_vk->allocate_buffer((size_t)RANK * max_out_dim * sizeof(float));
        ly.grad    = g_vk->allocate_buffer((size_t)RANK * max_out_dim * sizeof(float));
        ly.W_new   = g_vk->allocate_buffer((size_t)max_in_dim * max_out_dim * sizeof(float));

        ly.size_q = H * H;
        ly.size_k = H * H;
        ly.size_v = H * H;
        ly.size_o = H * H;
        ly.size_gate = H * 4*H;
        ly.size_up = H * 4*H;
        ly.size_down = 4*H * H;

        float scale_q = sqrt(2.0f / H);
        float scale_k = sqrt(2.0f / H);
        float scale_v = sqrt(2.0f / H);
        float scale_o = sqrt(1.0f / H);
        float scale_gate = sqrt(2.0f / H);
        float scale_up = sqrt(2.0f / H);
        float scale_down = sqrt(2.0f / (4*H));

        std::vector<float> q_data(H * H), k_data(H * H), v_data(H * H), o_data(H * H);
        std::vector<float> gate_data(H * 4*H), up_data(H * 4*H), down_data(4*H * H);
        std::vector<float> rms_a_data(H, 1.0f), rms_f_data(H, 1.0f);
        std::vector<float> R_data(RANK * H);

        for (auto& val : q_data) val = dist(rng) * scale_q;
        for (auto& val : k_data) val = dist(rng) * scale_k;
        for (auto& val : v_data) val = dist(rng) * scale_v;
        for (auto& val : o_data) val = dist(rng) * scale_o;
        for (auto& val : gate_data) val = dist(rng) * scale_gate;
        for (auto& val : up_data) val = dist(rng) * scale_up;
        for (auto& val : down_data) val = dist(rng) * scale_down;
        for (auto& val : R_data) val = dist(rng);

        upload(ly.q_proj, q_data.data(), H * H * sizeof(float));
        upload(ly.k_proj, k_data.data(), H * H * sizeof(float));
        upload(ly.v_proj, v_data.data(), H * H * sizeof(float));
        upload(ly.o_proj, o_data.data(), H * H * sizeof(float));
        upload(ly.gate_proj, gate_data.data(), H * 4*H * sizeof(float));
        upload(ly.up_proj, up_data.data(), H * 4*H * sizeof(float));
        upload(ly.down_proj, down_data.data(), 4*H * H * sizeof(float));
        upload(ly.rms_attn, rms_a_data.data(), H * sizeof(float));
        upload(ly.rms_ffn, rms_f_data.data(), H * sizeof(float));
        // R_proj теперь выделен под [RANK, 4H]; обнуляем и заполняем первые H столбцов
        zero_buffer(ly.R_proj);
        upload(ly.R_proj, R_data.data(), RANK * H * sizeof(float));

        zero_buffer(ly.XTX);
        zero_buffer(ly.XTY);
        zero_buffer(ly.W_small);
        zero_buffer(ly.grad);
        zero_buffer(ly.W_new);
        ly.token_count = 0;
    }

    // LM Head scratch buffers
    lm_head_XTX = g_vk->allocate_buffer(RANK * RANK * sizeof(float));
    lm_head_XTY = g_vk->allocate_buffer(RANK * H * sizeof(float));
    lm_head_W_small = g_vk->allocate_buffer(RANK * V * sizeof(float));
    lm_head_grad = g_vk->allocate_buffer(RANK * V * sizeof(float));
    lm_head_W_new = g_vk->allocate_buffer(V * H * sizeof(float));

    zero_buffer(lm_head_XTX);
    zero_buffer(lm_head_XTY);
    zero_buffer(lm_head_W_small);
    zero_buffer(lm_head_grad);
    zero_buffer(lm_head_W_new);
}

// ============================================================================
// ОБУЧЕНИЕ
// ============================================================================
extern "C" float model_train_step(const int* tokens, const int* targets, int seq_len, float lr) {
    static int global_step = 0;
    static uzagpt::GPUBuffer* logits = nullptr;
    if (!logits) logits = g_vk->allocate_buffer(V * sizeof(float));

    float total_loss = 0;

    for (int pos = 0; pos < seq_len; pos++) {
        // Forward pass
        forward_token(tokens[pos], pos, kv_cache_k, kv_cache_v, logits, true);

        // Вычисление loss (на CPU, только для логирования)
        if (cpu_buffer.size() < (size_t)V) cpu_buffer.resize(V);
        download(cpu_buffer.data(), logits, V * sizeof(float));

        // Softmax
        float mx = *std::max_element(cpu_buffer.begin(), cpu_buffer.begin() + V);
        float sum = 0;
        for (int i = 0; i < V; i++) {
            cpu_buffer[i] = std::exp(cpu_buffer[i] - mx);
            sum += cpu_buffer[i];
        }
        float prob = cpu_buffer[targets[pos]] / sum;
        if (prob < 1e-8f) prob = 1e-8f;
        total_loss -= std::log(prob);

        // TODO: Обновление статистик для loss (градиенты не нужны для FLETTOHM)
        global_step++;
    }

    // Обновление весов
    if (global_step % UPDATE_EVERY == 0) {
        for (int l = 0; l < L; l++) {
            auto& ly = g_layers[l];
            flettohm_update(ly.q_proj,   ly, H,   H);
            flettohm_update(ly.k_proj,   ly, H,   H);
            flettohm_update(ly.v_proj,   ly, H,   H);
            flettohm_update(ly.o_proj,   ly, H,   H);
            flettohm_update(ly.gate_proj, ly, 4*H, H);
            flettohm_update(ly.up_proj,   ly, 4*H, H);
            flettohm_update(ly.down_proj, ly, H,   4*H);
        }
        flettohm_update(lm_head, g_layers[0], V, H);
    }

    return total_loss / seq_len;
}

// ============================================================================
// ГЕНЕРАЦИЯ
// ============================================================================
extern "C" int model_generate(const int* prompt, int prompt_len,
                               int* out, int max_tokens,
                               float temp, float top_p) {
    // Очистка KV кэша
    for (int l = 0; l < L; l++) {
        zero_buffer(kv_cache_k[l]);
        zero_buffer(kv_cache_v[l]);
    }

    static uzagpt::GPUBuffer* logits = nullptr;
    if (!logits) logits = g_vk->allocate_buffer(V * sizeof(float));

    int pos = 0;
    int last_token = 0;

    // Обработка промпта
    for (int i = 0; i < prompt_len; i++) {
        forward_token(prompt[i], pos, kv_cache_k, kv_cache_v, logits, false);
        last_token = prompt[i];
        pos++;
    }

    // Генерация
    for (int gen = 0; gen < max_tokens; gen++) {
        forward_token(last_token, pos, kv_cache_k, kv_cache_v, logits, false);
        int next_token = sample_gpu(logits, V, temp, top_p);
        out[gen] = next_token;
        last_token = next_token;
        pos++;

        if (next_token == 1 || next_token == 2) break; // EOS
    }

    return max_tokens;
}

// ============================================================================
// ОЧИСТКА
// ============================================================================
extern "C" void model_cleanup() {
    g_vk->free_buffer(wte);
    g_vk->free_buffer(lm_head);
    g_vk->free_buffer(rms_final);

    for (auto& ly : g_layers) {
        g_vk->free_buffer(ly.q_proj);
        g_vk->free_buffer(ly.k_proj);
        g_vk->free_buffer(ly.v_proj);
        g_vk->free_buffer(ly.o_proj);
        g_vk->free_buffer(ly.gate_proj);
        g_vk->free_buffer(ly.up_proj);
        g_vk->free_buffer(ly.down_proj);
        g_vk->free_buffer(ly.rms_attn);
        g_vk->free_buffer(ly.rms_ffn);
        g_vk->free_buffer(ly.XTX);
        g_vk->free_buffer(ly.XTY);
        g_vk->free_buffer(ly.R_proj);
        g_vk->free_buffer(ly.temp_proj);
    }

    for (auto* buf : kv_cache_k) g_vk->free_buffer(buf);
    for (auto* buf : kv_cache_v) g_vk->free_buffer(buf);
}

// ============================================================================
// SAVE / LOAD — GPU↔CPU сериализация весов (бинарный формат)
//   Формат: magic, H, L, V, RANK, [wte, lm_head, rms_final],
//           для каждого слоя: q,k,v,o,gate,up,down,rms_attn,rms_ffn.
//   Буферы R_proj / W_small / grad / W_new / XTX / XTY / temp_proj
//   не персистим — они либо случайные, либо восстанавливаются из token_count.
// ============================================================================
static const uint32_t MODEL_MAGIC = 0x464C4554;  // 'FLET'

static bool write_buf(FILE* f, uzagpt::GPUBuffer* buf, size_t n_floats) {
    if (n_floats == 0) return true;
    cpu_buffer.resize(n_floats);
    download(cpu_buffer.data(), buf, n_floats * sizeof(float));
    return fwrite(cpu_buffer.data(), sizeof(float), n_floats, f) == n_floats;
}

static bool read_buf(FILE* f, uzagpt::GPUBuffer* buf, size_t n_floats) {
    if (n_floats == 0) return true;
    cpu_buffer.resize(n_floats);
    if (fread(cpu_buffer.data(), sizeof(float), n_floats, f) != n_floats) return false;
    upload(buf, cpu_buffer.data(), n_floats * sizeof(float));
    return true;
}

extern "C" int model_save(const char* path) {
    if (!path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t header[5] = { MODEL_MAGIC, (uint32_t)H, (uint32_t)L, (uint32_t)V, (uint32_t)RANK };
    if (fwrite(header, sizeof(uint32_t), 5, f) != 5) { fclose(f); return -1; }

    bool ok = true;
    ok &= write_buf(f, wte,      (size_t)V * H);
    ok &= write_buf(f, lm_head,  (size_t)V * H);
    ok &= write_buf(f, rms_final, (size_t)H);

    for (auto& ly : g_layers) {
        ok &= write_buf(f, ly.q_proj,    (size_t)H * H);
        ok &= write_buf(f, ly.k_proj,    (size_t)H * H);
        ok &= write_buf(f, ly.v_proj,    (size_t)H * H);
        ok &= write_buf(f, ly.o_proj,    (size_t)H * H);
        ok &= write_buf(f, ly.gate_proj, (size_t)H * 4 * H);
        ok &= write_buf(f, ly.up_proj,   (size_t)H * 4 * H);
        ok &= write_buf(f, ly.down_proj, (size_t)4 * H * H);
        ok &= write_buf(f, ly.rms_attn,  (size_t)H);
        ok &= write_buf(f, ly.rms_ffn,   (size_t)H);
    }
    fclose(f);
    return ok ? 0 : -1;
}

extern "C" int model_load(const char* path) {
    // Ленивая инициализация GPU-ресурсов: uzaleat_core не вызывает model_init
    // в chat-флоу, поэтому плагин сам отвечает за init.
    if (!wte) {
        // Компактные дефолты, безопасные для большинства GPU по памяти.
        // hidden=256, layers=4, vocab=4096, ctx=512, rank=32, update=512
        model_init(256, 4, 4096, 512, 32, 512);
    }

    if (!path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) {
        // Нет сохранённого состояния — сохраняем только что инициализированные случайные веса
        return model_save(path);
    }

    uint32_t header[5] = {};
    if (fread(header, sizeof(uint32_t), 5, f) != 5) { fclose(f); return -1; }
    if (header[0] != MODEL_MAGIC) { fclose(f); return -1; }
    if ((int)header[1] != H || (int)header[2] != L ||
        (int)header[3] != V || (int)header[4] != RANK) {
        fclose(f); return -1;
    }

    bool ok = true;
    ok &= read_buf(f, wte,      (size_t)V * H);
    ok &= read_buf(f, lm_head,  (size_t)V * H);
    ok &= read_buf(f, rms_final, (size_t)H);

    for (auto& ly : g_layers) {
        ok &= read_buf(f, ly.q_proj,    (size_t)H * H);
        ok &= read_buf(f, ly.k_proj,    (size_t)H * H);
        ok &= read_buf(f, ly.v_proj,    (size_t)H * H);
        ok &= read_buf(f, ly.o_proj,    (size_t)H * H);
        ok &= read_buf(f, ly.gate_proj, (size_t)H * 4 * H);
        ok &= read_buf(f, ly.up_proj,   (size_t)H * 4 * H);
        ok &= read_buf(f, ly.down_proj, (size_t)4 * H * H);
        ok &= read_buf(f, ly.rms_attn,  (size_t)H);
        ok &= read_buf(f, ly.rms_ffn,   (size_t)H);
    }
    fclose(f);
    return ok ? 0 : -1;
}
