// main.cpp — ane.cpp: Apple Neural Engine LLM inference tool
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <ane_lm/common.h>
#include <nlohmann/json.hpp>
#include "utils.h"
#include "generate.h"
#include "core/model_loader.h"
#include "core/sampling.h"

// ObjC autorelease pool via C runtime API
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s generate --model <path> [--prompt <text>] [options]\n", prog);
    fprintf(stderr, "  %s chat --model <path> [options]\n", prog);
    fprintf(stderr, "  %s convert --model <path>\n", prog);
    fprintf(stderr, "  %s serve --model <path> [options]\n", prog);
    fprintf(stderr, "\nSubcommands:\n");
    fprintf(stderr, "  generate    Single-shot text generation\n");
    fprintf(stderr, "  chat        Interactive multi-turn chat\n");
    fprintf(stderr, "  convert     Convert model weights from BF16 to FP16\n");
    fprintf(stderr, "  serve       Single-process concurrent generation server\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model <path>    Path to target model directory (required)\n");
    fprintf(stderr, "  --draft-model <path> Path to optional draft model directory\n");
    fprintf(stderr, "  --prompt <text>   Input prompt (generate only, default: \"Hello\")\n");
    fprintf(stderr, "  --max-tokens N    Max tokens per response (default: unlimited)\n");
    fprintf(stderr, "  --temp T          Temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-k N         Top-k sampling cutoff (default: 20, 0=off)\n");
    fprintf(stderr, "  --top-p P         Top-p nucleus cutoff (default: 0.8, 1.0=off)\n");
    fprintf(stderr, "  --repeat-penalty P Repetition penalty (default: 1.0, 1.0=off)\n");
    fprintf(stderr, "  --presence-penalty P Presence penalty (default: 1.5)\n");
    fprintf(stderr, "  --frequency-penalty P Frequency penalty (default: 0.0)\n");
    fprintf(stderr, "  --enable-thinking Enable thinking/reasoning mode\n");
    fprintf(stderr, "  --no-ane-cache    Disable persistent ANE compile cache\n");
    fprintf(stderr, "  --port N          Server port for serve mode (default: 8088)\n");
    fprintf(stderr, "  --sessions N      Session pool size for serve mode (default: 4)\n");
    fprintf(stderr, "  -v, --verbose     Show detailed initialization info\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s generate --model /path/to/Qwen3.5-0.8B --prompt \"Hello\" --max-tokens 50\n", prog);
    fprintf(stderr, "  %s chat --model /path/to/Qwen3.5-0.8B\n", prog);
}

struct Args {
    const char* model_dir = nullptr;
    const char* draft_model_dir = nullptr;
    const char* prompt = "Hello";
    float temperature = 0.7f;
    int max_tokens = 0;
    float repetition_penalty = 1.0f;
    float presence_penalty = 1.5f;
    float frequency_penalty = 0.0f;
    int top_k = 20;
    float top_p = 0.8f;
    bool ane_cache = true;
    bool enable_thinking = false;
    int port = 8088;
    int sessions = 4;
};

static Args parse_args(int argc, char* argv[], int start) {
    Args args;
    for (int i = start; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
            args.draft_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repetition_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--presence-penalty") == 0 && i + 1 < argc) {
            args.presence_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--frequency-penalty") == 0 && i + 1 < argc) {
            args.frequency_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            args.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            args.top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--enable-thinking") == 0) {
            args.enable_thinking = true;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            args.port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sessions") == 0 && i + 1 < argc) {
            args.sessions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-ane-cache") == 0) {
            args.ane_cache = false;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        }
    }
    return args;
}

static bool is_stop_token(int token, const Tokenizer& tokenizer) {
    return token == tokenizer.eos_id() || token == tokenizer.im_end_id();
}

static std::string format_prompt(Tokenizer& tokenizer, const std::string& prompt, bool enable_thinking) {
    std::vector<std::pair<std::string, std::string>> messages = {{"user", prompt}};
    if (tokenizer.has_chat_template()) {
        return tokenizer.apply_chat_template(messages, true, enable_thinking);
    }
    return prompt + "\n";
}

static std::string read_line_fd(int fd) {
    std::string line;
    char ch = 0;
    while (true) {
        ssize_t n = recv(fd, &ch, 1, 0);
        if (n <= 0) break;
        if (ch == '\n') break;
        line.push_back(ch);
        if (line.size() > (1u << 20)) break;
    }
    return line;
}

static void write_json_line(int fd, const nlohmann::json& payload) {
    std::string line = payload.dump();
    line.push_back('\n');
    const char* data = line.c_str();
    size_t left = line.size();
    while (left > 0) {
        ssize_t n = send(fd, data, left, 0);
        if (n <= 0) break;
        data += n;
        left -= (size_t)n;
    }
}

struct ServeRequest {
    int client_fd = -1;
    int id = 0;
    std::string prompt;
    int max_tokens = 0;
    bool enable_thinking = false;
    SamplingParams sampling = {};
    std::unique_ptr<Qwen35Model::Session> session;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    float* logits = nullptr;
    int sampler_vocab = 0;
    int prompt_token_count = 0;
    double prompt_tps = 0.0;
    Timer generation_timer;
    bool generation_timer_started = false;
};

static void finish_request(Tokenizer& tokenizer, ServeRequest& req, const char* error = nullptr) {
    nlohmann::json out;
    out["id"] = req.id;
    out["ok"] = (error == nullptr);
    if (error) {
        out["error"] = error;
    } else {
        out["text"] = tokenizer.decode(req.generated_tokens);
        out["prompt_tokens"] = req.prompt_token_count;
        out["prompt_tps"] = req.prompt_tps;
        out["generation_tokens"] = (int)req.generated_tokens.size();
        out["generation_tps"] = req.generated_tokens.empty()
            ? 0.0
            : req.generated_tokens.size() / (req.generation_timer.elapsed_ms() / 1000.0);
    }
    write_json_line(req.client_fd, out);
    close(req.client_fd);
    req.client_fd = -1;
}

static int cmd_serve(Qwen35Model& model, Tokenizer& tokenizer, const Args& args) {
    if (args.sessions < 1) {
        fprintf(stderr, "Error: --sessions must be >= 1\n");
        return 1;
    }
    if (args.port <= 0) {
        fprintf(stderr, "Error: --port must be > 0\n");
        return 1;
    }

    signal(SIGPIPE, SIG_IGN);

    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::unique_ptr<ServeRequest>> pending;
    std::vector<std::unique_ptr<ServeRequest>> active;
    std::vector<std::unique_ptr<Qwen35Model::Session>> available_sessions;
    for (int i = 0; i < args.sessions; i++) {
        auto session = model.create_session();
        if (!session) {
            fprintf(stderr, "Error: failed to allocate serve session %d\n", i + 1);
            return 1;
        }
        model.reset_session(*session);
        available_sessions.push_back(std::move(session));
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "socket failed: %s\n", strerror(errno));
        return 1;
    }
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)args.port);
    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "bind failed on port %d: %s\n", args.port, strerror(errno));
        close(server_fd);
        return 1;
    }
    if (listen(server_fd, 128) < 0) {
        fprintf(stderr, "listen failed: %s\n", strerror(errno));
        close(server_fd);
        return 1;
    }

    fprintf(stderr, "ane.cpp serve listening on 127.0.0.1:%d with %d sessions\n", args.port, args.sessions);

    std::thread scheduler([&] {
        while (true) {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return !pending.empty() || !active.empty(); });

            while (!pending.empty() && !available_sessions.empty()) {
                auto req = std::move(pending.front());
                pending.erase(pending.begin());
                req->session = std::move(available_sessions.back());
                available_sessions.pop_back();
                model.reset_session(*req->session);
                active.push_back(std::move(req));
            }

            if (active.empty()) {
                continue;
            }

            auto batch = std::move(active);
            active.clear();
            lock.unlock();

            std::vector<std::unique_ptr<ServeRequest>> next_active;
            std::vector<int> decode_tokens;
            std::vector<int> decode_positions;

            for (auto& req : batch) {
                if (!req->logits) {
                    std::string formatted = format_prompt(tokenizer, req->prompt, req->enable_thinking);
                    req->prompt_tokens = tokenizer.encode(formatted);
                    req->prompt_token_count = (int)req->prompt_tokens.size();
                    req->sampler_vocab = std::min(model.vocab_size(), tokenizer.vocab_size());
                    Timer prefill_timer;
                    req->logits = model.prefill(*req->session, req->prompt_tokens, 0);
                    if (!req->logits) {
                        finish_request(tokenizer, *req, "prefill failed");
                        std::lock_guard<std::mutex> done_lock(mu);
                        available_sessions.push_back(std::move(req->session));
                        continue;
                    }
                    req->prompt_tps = req->prompt_tokens.empty()
                        ? 0.0
                        : req->prompt_tokens.size() / (prefill_timer.elapsed_ms() / 1000.0);
                }

                int limit = (req->max_tokens > 0) ? req->max_tokens : INT_MAX;
                bool done = false;
                if ((int)req->generated_tokens.size() >= limit) {
                    done = true;
                } else {
                    if (!req->generation_timer_started) {
                        req->generation_timer.reset();
                        req->generation_timer_started = true;
                    }
                    int next_token = sample_token(req->logits, req->sampler_vocab, req->sampling, req->generated_tokens);
                    if (is_stop_token(next_token, tokenizer)) {
                        done = true;
                    } else {
                        req->generated_tokens.push_back(next_token);
                        if ((int)req->generated_tokens.size() < limit) {
                            decode_tokens.push_back(next_token);
                            decode_positions.push_back(req->prompt_token_count + (int)req->generated_tokens.size() - 1);
                        } else {
                            done = true;
                        }
                    }
                }

                if (done) {
                    finish_request(tokenizer, *req);
                    std::lock_guard<std::mutex> done_lock(mu);
                    available_sessions.push_back(std::move(req->session));
                } else {
                    next_active.push_back(std::move(req));
                }
            }

            if (!next_active.empty()) {
                bool decode_ok = true;
                if (next_active.size() == 1) {
                    auto& req = next_active[0];
                    req->logits = model.forward(*req->session, decode_tokens[0], decode_positions[0]);
                    decode_ok = (req->logits != nullptr);
                } else {
                    std::vector<Qwen35Model::Session*> sessions;
                    sessions.reserve(next_active.size());
                    for (auto& req : next_active) sessions.push_back(req->session.get());
                    decode_ok = model.forward_batch(sessions.data(), decode_tokens.data(), decode_positions.data(), (int)next_active.size());
                    if (decode_ok) {
                        for (auto& req : next_active) req->logits = req->session->logits;
                    } else {
                        fprintf(stderr, "Batched decode failed, falling back to sequential\n");
                        for (size_t i = 0; i < next_active.size(); i++) {
                            auto& req = next_active[i];
                            req->logits = model.forward(*req->session, decode_tokens[i], decode_positions[i]);
                            if (!req->logits) {
                                finish_request(tokenizer, *req, "generation failed");
                                std::lock_guard<std::mutex> done_lock(mu);
                                available_sessions.push_back(std::move(req->session));
                            }
                        }
                    }
                }

                std::lock_guard<std::mutex> done_lock(mu);
                for (auto& req : next_active) {
                    if (!req->session) continue;
                    if (!req->logits) {
                        finish_request(tokenizer, *req, "generation failed");
                        available_sessions.push_back(std::move(req->session));
                        continue;
                    }
                    active.push_back(std::move(req));
                }
            }

            cv.notify_all();
        }
    });
    scheduler.detach();

    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            fprintf(stderr, "accept failed: %s\n", strerror(errno));
            continue;
        }

        std::string line = read_line_fd(client_fd);
        if (line.empty()) {
            close(client_fd);
            continue;
        }

        nlohmann::json in = nlohmann::json::parse(line, nullptr, false);
        if (in.is_discarded() || !in.contains("prompt") || !in["prompt"].is_string()) {
            write_json_line(client_fd, {{"ok", false}, {"error", "invalid request"}});
            close(client_fd);
            continue;
        }

        auto req = std::make_unique<ServeRequest>();
        req->client_fd = client_fd;
        req->id = in.value("id", 0);
        req->prompt = in.value("prompt", std::string());
        req->max_tokens = in.value("max_tokens", args.max_tokens);
        req->enable_thinking = in.value("enable_thinking", args.enable_thinking);
        req->sampling.temperature = in.value("temp", args.temperature);
        req->sampling.repetition_penalty = in.value("repeat_penalty", args.repetition_penalty);
        req->sampling.presence_penalty = in.value("presence_penalty", args.presence_penalty);
        req->sampling.frequency_penalty = in.value("frequency_penalty", args.frequency_penalty);
        req->sampling.top_k = in.value("top_k", args.top_k);
        req->sampling.top_p = in.value("top_p", args.top_p);

        {
            std::lock_guard<std::mutex> lock(mu);
            pending.push_back(std::move(req));
        }
        cv.notify_all();
    }

    close(server_fd);
    return 0;
}

static int cmd_generate(LLMModel& model, Tokenizer& tokenizer,
                        DraftModelContext* draft, const Args& args) {
    LOG("Prompt: \"%s\"\n", args.prompt);

    SamplingParams sampling;
    sampling.temperature = args.temperature;
    sampling.repetition_penalty = args.repetition_penalty;
    sampling.presence_penalty = args.presence_penalty;
    sampling.frequency_penalty = args.frequency_penalty;
    sampling.top_k = args.top_k;
    sampling.top_p = args.top_p;

    GenerationResponse last{};
    bool first = true;
    stream_generate(model, tokenizer, std::string(args.prompt),
        args.max_tokens, args.enable_thinking, sampling,
        [&](const GenerationResponse& r) {
            if (r.token == -1) {
                last = r;
                return;
            }
            if (!r.text.empty()) {
                if (first) { fprintf(stderr, "==========\n"); first = false; }
                fprintf(stderr, "%s", r.text.c_str());
            }
            last = r;
        }, draft);

    fprintf(stderr, "\n==========\n");
    fprintf(stderr, "Prompt: %d tokens, %.3f tokens-per-sec\n",
            last.prompt_tokens, last.prompt_tps);
    fprintf(stderr, "Generation: %d tokens, %.3f tokens-per-sec\n",
            last.generation_tokens, last.generation_tps);
    return 0;
}

static int cmd_chat(LLMModel& model, Tokenizer& tokenizer,
                    DraftModelContext* draft, const Args& args) {
    std::vector<std::pair<std::string, std::string>> messages;
    char buf[4096];

    while (true) {
        fprintf(stderr, ">>> ");
        if (!fgets(buf, sizeof(buf), stdin)) {
            // EOF (Ctrl-D)
            fprintf(stderr, "\n");
            break;
        }

        // Strip trailing newline
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

        // Skip empty input
        if (buf[0] == '\0') continue;

        // Exit commands
        if (strcmp(buf, "/bye") == 0 || strcmp(buf, "/exit") == 0) break;

        // Add user message
        messages.push_back({"user", std::string(buf)});

        // Reset model state and generate with full history
        model.reset();
        if (draft && draft->model) draft->model->reset();

        SamplingParams sampling;
        sampling.temperature = args.temperature;
        sampling.repetition_penalty = args.repetition_penalty;
        sampling.presence_penalty = args.presence_penalty;
        sampling.frequency_penalty = args.frequency_penalty;
        sampling.top_k = args.top_k;
        sampling.top_p = args.top_p;

        std::string assistant_text;
        GenerationResponse last{};
        stream_generate(model, tokenizer, messages,
            args.max_tokens, args.enable_thinking, sampling,
            [&](const GenerationResponse& r) {
                if (r.token == -1) {
                    last = r;
                    return;
                }
                if (!r.text.empty()) {
                    fprintf(stderr, "%s", r.text.c_str());
                    assistant_text += r.text;
                }
                last = r;
            }, draft);

        fprintf(stderr, "\n");

        // Add assistant response to history
        messages.push_back({"assistant", assistant_text});

        fprintf(stderr, "[%d prompt tokens, %.1f t/s | %d gen tokens, %.1f t/s]\n\n",
                last.prompt_tokens, last.prompt_tps,
                last.generation_tokens, last.generation_tps);
    }

    return 0;
}

static int cmd_convert(const Args& args) {
    std::string model_dir = args.model_dir;

    // Discover all safetensors files (single-file or sharded) and convert them.
    auto weights = ModelWeights::open(model_dir);
    if (!weights) {
        fprintf(stderr, "Error: failed to load model weights in %s\n", model_dir.c_str());
        return 1;
    }

    std::string output_dir = model_dir + "/ane_weights";

    Timer timer;
    int written = weights->write_ane_blobs(output_dir);
    double elapsed = timer.elapsed_ms();

    if (written < 0) {
        fprintf(stderr, "Error: conversion failed\n");
        return 1;
    }

    fprintf(stderr, "Done in %.1f ms\n", elapsed);
    return 0;
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    srand48(time(nullptr));
    setbuf(stdout, nullptr);

    // Need at least a subcommand
    if (argc < 2) {
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // Check for --help before subcommand
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 0;
    }

    // Determine subcommand
    const char* subcmd = argv[1];
    bool is_generate = (strcmp(subcmd, "generate") == 0);
    bool is_chat = (strcmp(subcmd, "chat") == 0);
    bool is_convert = (strcmp(subcmd, "convert") == 0);
    bool is_serve = (strcmp(subcmd, "serve") == 0);

    if (!is_generate && !is_chat && !is_convert && !is_serve) {
        fprintf(stderr, "Unknown subcommand: %s\n\n", subcmd);
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // Parse args after subcommand
    Args args = parse_args(argc, argv, 2);

    if (!args.model_dir) {
        fprintf(stderr, "Error: --model is required\n\n");
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // convert doesn't need model/tokenizer loading
    if (is_convert) {
        int ret = cmd_convert(args);
        objc_autoreleasePoolPop(pool);
        return ret;
    }

    LOG("=== ane.cpp: Apple Neural Engine LLM Inference ===\n");
    LOG("Model: %s\n", args.model_dir);
    if (args.draft_model_dir) LOG("Draft model: %s\n", args.draft_model_dir);
    LOG("Mode: %s\n", is_chat ? "chat" : (is_serve ? "serve" : "generate"));
    LOG("Temperature: %.2f, Max tokens: %d\n", args.temperature, args.max_tokens);
    LOG("ANE compile cache: %s\n", args.ane_cache ? "enabled" : "disabled");

    // Load model + tokenizer
    std::unique_ptr<LLMModel> model;
    Tokenizer tokenizer;
    std::unique_ptr<LLMModel> draft_model;
    Tokenizer draft_tokenizer;
    DraftModelContext draft_ctx{};
    try {
        auto result = load(args.model_dir, args.ane_cache);
        model = std::move(result.first);
        tokenizer = std::move(result.second);
        if (args.draft_model_dir) {
            auto draft_result = load(args.draft_model_dir, args.ane_cache);
            draft_model = std::move(draft_result.first);
            draft_tokenizer = std::move(draft_result.second);
            draft_ctx.model = draft_model.get();
            draft_ctx.tokenizer = &draft_tokenizer;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    int ret;
    DraftModelContext* draft_ptr = draft_ctx.model ? &draft_ctx : nullptr;
    if (is_chat) {
        ret = cmd_chat(*model, tokenizer, draft_ptr, args);
    } else if (is_serve) {
        auto* qwen35 = dynamic_cast<Qwen35Model*>(model.get());
        if (!qwen35) {
            fprintf(stderr, "Error: serve currently supports Qwen3.5 models only\n");
            ret = 1;
        } else {
            ret = cmd_serve(*qwen35, tokenizer, args);
        }
    } else {
        ret = cmd_generate(*model, tokenizer, draft_ptr, args);
    }

    objc_autoreleasePoolPop(pool);
    return ret;
}
