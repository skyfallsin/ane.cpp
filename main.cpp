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
#include <atomic>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <ane_lm/common.h>
#include <nlohmann/json.hpp>
#include "utils.h"
#include "generate.h"
#include "core/model_loader.h"
#include "core/weight_cache.h"
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


// =====================================================
// HTTP helpers for OpenAI-compatible serve mode
// =====================================================

static std::string generate_request_id() {
    static std::atomic<int> counter{0};
    int c = counter.fetch_add(1);
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-ane-%d-%d", (int)time(nullptr), c);
    return buf;
}

static std::string extract_model_name(const char* model_dir) {
    std::string path(model_dir);
    while (!path.empty() && path.back() == '/') path.pop_back();
    auto pos = path.rfind('/');
    if (pos != std::string::npos) return path.substr(pos + 1);
    return path;
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    bool valid = false;
};

static HttpRequest parse_http_request(int fd) {
    HttpRequest req;
    std::string hdr;
    // Read until \r\n\r\n
    while (hdr.size() < 65536) {
        char ch;
        ssize_t n = recv(fd, &ch, 1, 0);
        if (n <= 0) return req;
        hdr.push_back(ch);
        if (hdr.size() >= 4 &&
            hdr[hdr.size()-4] == '\r' && hdr[hdr.size()-3] == '\n' &&
            hdr[hdr.size()-2] == '\r' && hdr[hdr.size()-1] == '\n') break;
    }

    // Parse request line: "POST /v1/chat/completions HTTP/1.1\r\n"
    auto first_crlf = hdr.find("\r\n");
    if (first_crlf == std::string::npos) return req;
    std::string request_line = hdr.substr(0, first_crlf);
    auto sp1 = request_line.find(' ');
    if (sp1 == std::string::npos) return req;
    auto sp2 = request_line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return req;
    req.method = request_line.substr(0, sp1);
    req.path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Find Content-Length (case-insensitive)
    int content_length = 0;
    std::string hdr_lower = hdr;
    for (auto& c : hdr_lower) c = (char)tolower((unsigned char)c);
    auto cl_pos = hdr_lower.find("content-length:");
    if (cl_pos != std::string::npos) {
        content_length = atoi(hdr.c_str() + cl_pos + 15);
    }

    // Read body
    if (content_length > 0 && content_length < (1 << 20)) {
        req.body.resize((size_t)content_length);
        size_t got = 0;
        while (got < (size_t)content_length) {
            ssize_t r = recv(fd, &req.body[got], (size_t)content_length - got, 0);
            if (r <= 0) break;
            got += (size_t)r;
        }
        req.body.resize(got);
    }

    req.valid = true;
    return req;
}

static void send_raw(int fd, const std::string& data) {
    const char* p = data.c_str();
    size_t left = data.size();
    while (left > 0) {
        ssize_t n = send(fd, p, left, 0);
        if (n <= 0) break;
        p += n;
        left -= (size_t)n;
    }
}

static void send_http_json(int fd, int status, const char* status_text,
                           const std::string& body) {
    std::string resp = "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" + body;
    send_raw(fd, resp);
}

static void send_sse_headers(int fd) {
    send_raw(fd, "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n");
}

static void send_sse_event(int fd, const nlohmann::json& data) {
    std::string event = "data: " + data.dump() + "\n\n";
    send_raw(fd, event);
}

static void send_sse_done(int fd) {
    send_raw(fd, "data: [DONE]\n\n");
}

// =====================================================
// OpenAI-compatible ServeRequest
// =====================================================

struct ServeRequest {
    int client_fd = -1;
    std::string request_id;
    std::string model_name;
    std::vector<std::pair<std::string, std::string>> messages;
    int max_tokens = 0;
    bool enable_thinking = false;
    bool stream = true;
    SamplingParams sampling = {};
    int64_t created = 0;

    // Session & generation state
    std::unique_ptr<Qwen35Model::Session> session;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    float* logits = nullptr;
    int sampler_vocab = 0;
    int prompt_token_count = 0;
    double prompt_tps = 0.0;
    Timer generation_timer;
    bool generation_timer_started = false;

    // Incremental text decode state (for streaming)
    std::string prev_decoded;
    std::string emitted_text;
    bool has_prev_decoded = false;
    bool sse_headers_sent = false;
    bool role_sent = false;
};

static bool is_utf8_continuation(uint8_t b) {
    return (b & 0xC0u) == 0x80u;
}

// Emit one token to client (streaming or accumulate for non-streaming)
static void emit_token_to_client(Tokenizer& tokenizer, ServeRequest& req, int token) {
    req.generated_tokens.push_back(token);
    std::string current_decoded = tokenizer.decode(req.generated_tokens);

    std::string piece;
    if (req.has_prev_decoded) {
        // Find stable prefix between prev and current decode
        size_t lcp = std::min(req.prev_decoded.size(), current_decoded.size());
        size_t i = 0;
        while (i < lcp && req.prev_decoded[i] == current_decoded[i]) i++;
        // Back up to UTF-8 boundary
        while (i > 0 && is_utf8_continuation((uint8_t)req.prev_decoded[i])) i--;
        std::string stable = req.prev_decoded.substr(0, i);
        if (stable.size() >= req.emitted_text.size() &&
            stable.compare(0, req.emitted_text.size(), req.emitted_text) == 0) {
            piece = stable.substr(req.emitted_text.size());
            req.emitted_text = std::move(stable);
        }
    }
    req.prev_decoded = std::move(current_decoded);
    req.has_prev_decoded = true;

    if (req.stream && !piece.empty()) {
        if (!req.sse_headers_sent) {
            send_sse_headers(req.client_fd);
            req.sse_headers_sent = true;
        }
        // Send role delta on first content
        if (!req.role_sent) {
            nlohmann::json role_chunk = {
                {"id", req.request_id},
                {"object", "chat.completion.chunk"},
                {"created", req.created},
                {"model", req.model_name},
                {"choices", {{
                    {"index", 0},
                    {"delta", {{"role", "assistant"}}},
                    {"finish_reason", nullptr}
                }}}
            };
            send_sse_event(req.client_fd, role_chunk);
            req.role_sent = true;
        }
        nlohmann::json chunk = {
            {"id", req.request_id},
            {"object", "chat.completion.chunk"},
            {"created", req.created},
            {"model", req.model_name},
            {"choices", {{
                {"index", 0},
                {"delta", {{"content", piece}}},
                {"finish_reason", nullptr}
            }}}
        };
        send_sse_event(req.client_fd, chunk);
    }
}

// Flush remaining decoded text (BPE can buffer partial codepoints)
static void flush_remaining_text(ServeRequest& req) {
    if (!req.has_prev_decoded) return;
    std::string tail;
    if (req.prev_decoded.size() > req.emitted_text.size() &&
        req.prev_decoded.compare(0, req.emitted_text.size(), req.emitted_text) == 0) {
        tail = req.prev_decoded.substr(req.emitted_text.size());
    }
    if (req.stream && !tail.empty()) {
        if (!req.sse_headers_sent) {
            send_sse_headers(req.client_fd);
            req.sse_headers_sent = true;
        }
        if (!req.role_sent) {
            nlohmann::json role_chunk = {
                {"id", req.request_id},
                {"object", "chat.completion.chunk"},
                {"created", req.created},
                {"model", req.model_name},
                {"choices", {{
                    {"index", 0},
                    {"delta", {{"role", "assistant"}}},
                    {"finish_reason", nullptr}
                }}}
            };
            send_sse_event(req.client_fd, role_chunk);
            req.role_sent = true;
        }
        nlohmann::json chunk = {
            {"id", req.request_id},
            {"object", "chat.completion.chunk"},
            {"created", req.created},
            {"model", req.model_name},
            {"choices", {{
                {"index", 0},
                {"delta", {{"content", tail}}},
                {"finish_reason", nullptr}
            }}}
        };
        send_sse_event(req.client_fd, chunk);
    }
    req.emitted_text = req.prev_decoded;
}

static void finish_request(Tokenizer& tokenizer, ServeRequest& req, const char* error = nullptr) {
    double gen_tps = req.generated_tokens.empty()
        ? 0.0
        : req.generated_tokens.size() / (req.generation_timer.elapsed_ms() / 1000.0);

    if (error) {
        if (req.stream && req.sse_headers_sent) {
            // Error mid-stream: send error event and close
            nlohmann::json err_chunk = {
                {"error", {{"message", error}, {"type", "server_error"}}}
            };
            send_sse_event(req.client_fd, err_chunk);
            send_sse_done(req.client_fd);
        } else {
            nlohmann::json err_resp = {
                {"error", {{"message", error}, {"type", "server_error"}, {"code", 500}}}
            };
            send_http_json(req.client_fd, 500, "Internal Server Error", err_resp.dump());
        }
    } else if (req.stream) {
        flush_remaining_text(req);
        if (!req.sse_headers_sent) {
            send_sse_headers(req.client_fd);
            req.sse_headers_sent = true;
        }
        // Final chunk with finish_reason
        nlohmann::json final_chunk = {
            {"id", req.request_id},
            {"object", "chat.completion.chunk"},
            {"created", req.created},
            {"model", req.model_name},
            {"choices", {{
                {"index", 0},
                {"delta", nlohmann::json::object()},
                {"finish_reason", "stop"}
            }}},
            {"usage", {
                {"prompt_tokens", req.prompt_token_count},
                {"completion_tokens", (int)req.generated_tokens.size()},
                {"total_tokens", req.prompt_token_count + (int)req.generated_tokens.size()}
            }}
        };
        send_sse_event(req.client_fd, final_chunk);
        send_sse_done(req.client_fd);
    } else {
        // Non-streaming: send complete response
        std::string full_text = req.generated_tokens.empty()
            ? "" : tokenizer.decode(req.generated_tokens);
        nlohmann::json resp = {
            {"id", req.request_id},
            {"object", "chat.completion"},
            {"created", req.created},
            {"model", req.model_name},
            {"choices", {{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", full_text}}},
                {"finish_reason", "stop"}
            }}},
            {"usage", {
                {"prompt_tokens", req.prompt_token_count},
                {"completion_tokens", (int)req.generated_tokens.size()},
                {"total_tokens", req.prompt_token_count + (int)req.generated_tokens.size()}
            }}
        };
        send_http_json(req.client_fd, 200, "OK", resp.dump());
    }

    fprintf(stderr, "[%s] %d prompt (%.1f t/s) + %d gen (%.1f t/s)\n",
            req.request_id.c_str(), req.prompt_token_count, req.prompt_tps,
            (int)req.generated_tokens.size(), gen_tps);

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

    std::string model_name = extract_model_name(args.model_dir);
    int64_t server_created = (int64_t)time(nullptr);

    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::unique_ptr<ServeRequest>> pending;
    std::vector<std::unique_ptr<ServeRequest>> active;
    fprintf(stderr, "[boot] allocating %d sessions...\n", args.sessions);
    std::vector<std::unique_ptr<Qwen35Model::Session>> available_sessions;
    for (int i = 0; i < args.sessions; i++) {
        fprintf(stderr, "[boot] creating session %d/%d...\n", i + 1, args.sessions);
        auto session = model.create_session();
        if (!session) {
            fprintf(stderr, "Error: failed to allocate serve session %d\n", i + 1);
            return 1;
        }
        model.reset_session(*session);
        available_sessions.push_back(std::move(session));
        fprintf(stderr, "[boot] session %d/%d ready\n", i + 1, args.sessions);
    }

    fprintf(stderr, "[boot] creating socket...\n");
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

    fprintf(stderr, "[boot] socket bound and listening\n");
    fprintf(stderr, "ane.cpp serve: OpenAI-compatible API at http://127.0.0.1:%d/v1\n", args.port);
    fprintf(stderr, "  model: %s, sessions: %d\n", model_name.c_str(), args.sessions);

    fprintf(stderr, "[boot] starting scheduler thread...\n");
    // Scheduler thread — batched decode with per-token streaming
    std::thread scheduler([&] {
        fprintf(stderr, "[boot] scheduler thread running\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return !pending.empty() || !active.empty(); });
            fprintf(stderr, "[sched] woke: pending=%zu active=%zu avail=%zu\n",
                    pending.size(), active.size(), available_sessions.size());

            while (!pending.empty() && !available_sessions.empty()) {
                auto req = std::move(pending.front());
                pending.erase(pending.begin());
                req->session = std::move(available_sessions.back());
                available_sessions.pop_back();
                model.reset_session(*req->session);
                active.push_back(std::move(req));
            }

            if (active.empty()) {
                fprintf(stderr, "[sched] no active requests, waiting...\n");
                continue;
            }

            fprintf(stderr, "[sched] processing batch of %zu\n", active.size());
            auto batch = std::move(active);
            active.clear();
            lock.unlock();

            std::vector<std::unique_ptr<ServeRequest>> next_active;
            std::vector<int> decode_tokens;
            std::vector<int> decode_positions;

            for (auto& req : batch) {
                if (!req->logits) {
                    // Prefill: apply chat template to messages
                    std::string formatted;
                    if (tokenizer.has_chat_template()) {
                        formatted = tokenizer.apply_chat_template(
                            req->messages, true, req->enable_thinking);
                    } else {
                        for (auto& [role, content] : req->messages) {
                            (void)role;
                            formatted += content + "\n";
                        }
                    }
                    req->prompt_tokens = tokenizer.encode(formatted);
                    // Truncate to fit context: keep last (max_context - max_gen) tokens
                    int max_context = 8192;
                    int reserve_for_gen = (req->max_tokens > 0)
                        ? std::min(req->max_tokens, max_context / 2) : max_context / 2;
                    int max_prompt = max_context - reserve_for_gen;
                    if ((int)req->prompt_tokens.size() > max_prompt) {
                        int drop = (int)req->prompt_tokens.size() - max_prompt;
                        fprintf(stderr, "[sched] truncating %s: %d -> %d tokens (dropped %d from front)\n",
                                req->request_id.c_str(), (int)req->prompt_tokens.size(), max_prompt, drop);
                        req->prompt_tokens.erase(req->prompt_tokens.begin(),
                                                 req->prompt_tokens.begin() + drop);
                    }
                    req->prompt_token_count = (int)req->prompt_tokens.size();
                    req->sampler_vocab = std::min(model.vocab_size(), tokenizer.vocab_size());
                    fprintf(stderr, "[sched] prefilling %s: %d tokens...\n",
                            req->request_id.c_str(), (int)req->prompt_tokens.size());
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
                    int next_token = sample_token(req->logits, req->sampler_vocab,
                                                  req->sampling, req->generated_tokens);
                    if (is_stop_token(next_token, tokenizer)) {
                        done = true;
                    } else {
                        emit_token_to_client(tokenizer, *req, next_token);
                        if ((int)req->generated_tokens.size() < limit) {
                            decode_tokens.push_back(next_token);
                            decode_positions.push_back(
                                req->prompt_token_count +
                                (int)req->generated_tokens.size() - 1);
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
                    req->logits = model.forward(*req->session,
                                                decode_tokens[0], decode_positions[0]);
                    decode_ok = (req->logits != nullptr);
                } else {
                    std::vector<Qwen35Model::Session*> sessions;
                    sessions.reserve(next_active.size());
                    for (auto& req : next_active)
                        sessions.push_back(req->session.get());
                    decode_ok = model.forward_batch(
                        sessions.data(), decode_tokens.data(),
                        decode_positions.data(), (int)next_active.size());
                    if (decode_ok) {
                        for (auto& req : next_active)
                            req->logits = req->session->logits;
                    } else {
                        fprintf(stderr, "Batched decode failed, falling back to sequential\n");
                        for (size_t i = 0; i < next_active.size(); i++) {
                            auto& req = next_active[i];
                            req->logits = model.forward(*req->session,
                                                        decode_tokens[i], decode_positions[i]);
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
    fprintf(stderr, "[boot] entering accept loop\n");

    // Accept loop — parse HTTP requests and route
    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            fprintf(stderr, "accept failed: %s\n", strerror(errno));
            continue;
        }
        fprintf(stderr, "[http] accepted connection fd=%d\n", client_fd);

        std::thread([client_fd, &mu, &cv, &pending, &args, &model_name,
                     server_created] {
            HttpRequest http = parse_http_request(client_fd);
            fprintf(stderr, "[http] parsed: valid=%d method=%s path=%s body=%zu bytes\n",
                    http.valid, http.method.c_str(), http.path.c_str(), http.body.size());
            if (!http.valid) {
                fprintf(stderr, "[http] invalid request, closing fd=%d\n", client_fd);
                close(client_fd);
                return;
            }

            // CORS preflight
            if (http.method == "OPTIONS") {
                send_raw(client_fd, "HTTP/1.1 204 No Content\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                    "Access-Control-Max-Age: 86400\r\n"
                    "Connection: close\r\n\r\n");
                close(client_fd);
                return;
            }

            // GET /v1/models
            if (http.method == "GET" && http.path == "/v1/models") {
                nlohmann::json resp = {
                    {"object", "list"},
                    {"data", {{
                        {"id", model_name},
                        {"object", "model"},
                        {"created", server_created},
                        {"owned_by", "ane.cpp"}
                    }}}
                };
                send_http_json(client_fd, 200, "OK", resp.dump());
                close(client_fd);
                return;
            }

            // POST /v1/chat/completions
            if (http.method == "POST" && http.path == "/v1/chat/completions") {
                fprintf(stderr, "[http] chat/completions request, parsing body...\n");
                auto body = nlohmann::json::parse(http.body, nullptr, false);
                if (body.is_discarded() || !body.contains("messages")) {
                    fprintf(stderr, "[http] invalid body: discarded=%d has_messages=%d\n",
                            body.is_discarded(), body.contains("messages"));
                    nlohmann::json err = {
                        {"error", {{"message", "Invalid request: 'messages' required"},
                                   {"type", "invalid_request_error"}}}
                    };
                    send_http_json(client_fd, 400, "Bad Request", err.dump());
                    close(client_fd);
                    return;
                }

                auto req = std::make_unique<ServeRequest>();
                req->client_fd = client_fd;
                req->request_id = generate_request_id();
                req->model_name = model_name;
                req->created = (int64_t)time(nullptr);
                req->stream = body.value("stream", true);

                // Parse messages — content can be string or array of content blocks
                for (auto& msg : body["messages"]) {
                    std::string role = msg.value("role", "user");
                    std::string content;
                    if (msg.contains("content")) {
                        if (msg["content"].is_string()) {
                            content = msg["content"].get<std::string>();
                        } else if (msg["content"].is_array()) {
                            // OpenAI multi-modal format: [{"type":"text","text":"..."},...]
                            for (auto& part : msg["content"]) {
                                if (part.is_object() && part.value("type", "") == "text") {
                                    if (!content.empty()) content += "\n";
                                    content += part.value("text", "");
                                }
                            }
                        }
                    }
                    // Map "developer" -> "system" for compatibility
                    if (role == "developer") role = "system";
                    req->messages.emplace_back(std::move(role), std::move(content));
                }

                // Sampling parameters
                req->max_tokens = body.value("max_tokens",
                                  body.value("max_completion_tokens", args.max_tokens));
                req->sampling.temperature = body.value("temperature", args.temperature);
                req->sampling.top_p = body.value("top_p", args.top_p);
                req->sampling.presence_penalty = body.value("presence_penalty",
                                                            args.presence_penalty);
                req->sampling.frequency_penalty = body.value("frequency_penalty",
                                                             args.frequency_penalty);
                req->sampling.repetition_penalty = body.value("repetition_penalty",
                                                              args.repetition_penalty);
                if (body.contains("top_k")) {
                    req->sampling.top_k = body.value("top_k", args.top_k);
                } else {
                    req->sampling.top_k = args.top_k;
                }

                fprintf(stderr, "[http] parsed %zu messages, stream=%d, max_tokens=%d\n",
                        req->messages.size(), req->stream, req->max_tokens);

                // Enable thinking: check multiple conventions
                req->enable_thinking = args.enable_thinking;
                if (body.contains("enable_thinking")) {
                    req->enable_thinking = body.value("enable_thinking", false);
                } else if (body.contains("chat_template_kwargs")) {
                    auto& kwargs = body["chat_template_kwargs"];
                    if (kwargs.contains("enable_thinking")) {
                        req->enable_thinking = kwargs.value("enable_thinking", false);
                    }
                }

                fprintf(stderr, "[http] queuing request %s thinking=%d\n",
                        req->request_id.c_str(), req->enable_thinking);
                {
                    std::lock_guard<std::mutex> lock(mu);
                    pending.push_back(std::move(req));
                }
                cv.notify_all();
                return;
            }

            // 404 for everything else
            nlohmann::json err = {
                {"error", {{"message", "Not found: " + http.path},
                           {"type", "invalid_request_error"}}}
            };
            send_http_json(client_fd, 404, "Not Found", err.dump());
            close(client_fd);
        }).detach();
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
        fprintf(stderr, "Error: ANE blob conversion failed\n");
        return 1;
    }

    fprintf(stderr, "ANE blobs done in %.1f ms\n", elapsed);

    // Also build the f16 weight cache for zero-copy GPU loading
    fprintf(stderr, "Building f16 weight cache...\n");
    Timer cache_timer;
    if (ane_lm::WeightCache::build(model_dir, weights.get())) {
        fprintf(stderr, "f16 cache built in %.1f ms\n", cache_timer.elapsed_ms());
    } else {
        fprintf(stderr, "Warning: f16 cache build failed (GPU zero-copy will be unavailable)\n");
    }

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
