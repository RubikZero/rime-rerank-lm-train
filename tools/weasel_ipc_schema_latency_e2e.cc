// Measures the visible latency of the LM rerank against a *running* server --
// by default the installed production one, since the pipe name is only
// overridden when RIME_WEASEL_PIPE_NAME is set.
//
// The schema of the server is per session, and the plugin is only wired into
// the schemas that list it, so a measurement of the LM path has to select that
// schema *inside the session it measures*: the switcher hotkey (Control+grave)
// opens the schema/option menu and its item digit selects an entry.  The
// session is ended after each run, so the switch is repeated per run.
//
// Usage: weasel_ipc_schema_latency_e2e.exe <context> <keys> <key_delay_ms>
//        <poll_ms> <max_ms> <runs> [schema_digit]
//
// schema_digit = 0 (or omitted) leaves the session's schema alone.
#include <WeaselIPC.h>

#include <boost/interprocess/streams/bufferstream.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr UINT kControlMask = 1u << 2;  // ibus::CONTROL_MASK
constexpr UINT kGrave = 0x60;           // "Control+grave" as librime spells it

bool ReadResponse(LPWSTR buffer, UINT length, LPWSTR destination) {
  boost::interprocess::wbufferstream stream(buffer, length);
  stream.read(destination, WEASEL_IPC_BUFFER_LENGTH);
  return stream.good();
}

std::string ToUtf8(const std::wstring& value) {
  if (value.empty())
    return {};
  const int length = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                         static_cast<int>(value.size()), NULL,
                                         0, NULL, NULL);
  std::string result(length, '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), length, NULL, NULL);
  return result;
}

std::wstring ToWide(const char* value) {
  const int length = MultiByteToWideChar(CP_UTF8, 0, value, -1, NULL, 0);
  std::wstring result(length > 0 ? length - 1 : 0, L'\0');
  if (length > 0)
    MultiByteToWideChar(CP_UTF8, 0, value, -1, result.data(), length);
  return result;
}

std::string Fetch(weasel::Client& client) {
  WCHAR buffer[WEASEL_IPC_BUFFER_LENGTH] = {0};
  if (!client.GetResponseData(
          std::bind(ReadResponse, std::placeholders::_1,
                    std::placeholders::_2, buffer)))
    return {};
  return ToUtf8(buffer);
}

std::string CandidateSignature(const std::string& response) {
  const std::string key = "ctx.cand=";
  const size_t at = response.find(key);
  if (at == std::string::npos)
    return {};
  const size_t end = response.find('\n', at);
  return response.substr(at, end == std::string::npos ? std::string::npos
                                                      : end - at);
}

bool HasLmMark(const std::string& response) {
  return response.find("[LM]") != std::string::npos;
}

bool LmPending(const std::string& response) {
  const std::string key = "config.lm_pending=";
  const size_t at = response.find(key);
  if (at == std::string::npos)
    return true;  // unknown: keep polling
  return response.compare(at + key.size(), 1, "1") == 0;
}

std::string SchemaId(const std::string& response) {
  const std::string key = "status.schema_id=";
  const size_t at = response.find(key);
  if (at == std::string::npos)
    return "?";
  const size_t end = response.find('\n', at);
  return response.substr(at + key.size(), end == std::string::npos
                                              ? std::string::npos
                                              : end - at - key.size());
}

long long ElapsedMs(const std::chrono::steady_clock::time_point& from) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - from)
      .count();
}

// The line of the response that carries the visible candidates (and, for the
// switcher, the schema names).
void PrintCandidates(const std::string& label, const std::string& response) {
  const std::string key = "ctx.cand=";
  const size_t at = response.find(key);
  std::printf("%s: %s\n", label.c_str(),
              at == std::string::npos
                  ? "(no candidate line)"
                  : response.substr(at, response.find('\n', at) - at).c_str());
}

// Control+grave opens the switcher, the item digit picks a schema.
std::string SelectSchema(weasel::Client& client, int digit) {
  client.ProcessKeyEvent(weasel::KeyEvent(kGrave, kControlMask));
  const std::string menu = Fetch(client);
  PrintCandidates("MENU", menu);
  client.ProcessKeyEvent(weasel::KeyEvent(static_cast<UINT>('0' + digit), 0));
  const std::string picked = Fetch(client);
  PrintCandidates("PICKED", picked);
  if (menu.find("ctx.cand=") == std::string::npos)
    std::fprintf(stderr, "warning: the switcher menu did not open\n");
  return SchemaId(picked);
}

}  // namespace

int main(int argc, char** argv) {
  const char* context = argc > 1 ? argv[1] : "meeting at three pm, ";
  const char* keys = argc > 2 ? argv[2] : "beijing";
  const int key_delay_ms = argc > 3 ? std::atoi(argv[3]) : 150;
  const int poll_ms = argc > 4 ? std::atoi(argv[4]) : 120;
  const int max_ms = argc > 5 ? std::atoi(argv[5]) : 8000;
  const int runs = argc > 6 ? std::atoi(argv[6]) : 3;
  const int schema_digit = argc > 7 ? std::atoi(argv[7]) : 0;

  weasel::Client client;
  if (!client.Connect()) {
    std::fprintf(stderr, "cannot connect to WeaselServer\n");
    return 2;
  }

  std::vector<long long> visible_ms;
  std::vector<int> poll_counts;
  std::string schema_seen = "?";
  int warmup_failed = 0;

  for (int run = 0; run < runs; ++run) {
    client.StartSession();
    if (!client.Echo()) {
      if (++warmup_failed > 6)
        return 2;
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      --run;
      continue;
    }
    if (schema_digit > 0)
      schema_seen = SelectSchema(client, schema_digit);
    client.FocusIn();
    const std::string run_context =
        std::string(context) + " run" + std::to_string(run);
    client.SetSurroundingText(ToWide(run_context.c_str()));

    std::string baseline;
    for (const char* key = keys; *key; ++key) {
      client.ProcessKeyEvent(weasel::KeyEvent(*key, 0));
      baseline = Fetch(client);
      if (key_delay_ms > 0 && *(key + 1))
        std::this_thread::sleep_for(std::chrono::milliseconds(key_delay_ms));
    }
    if (schema_seen == "?")
      schema_seen = SchemaId(baseline);
    const std::string baseline_signature = CandidateSignature(baseline);
    const bool baseline_marked = HasLmMark(baseline);

    const auto start = std::chrono::steady_clock::now();
    long long seen_at = -1;
    int polls = 0;
    while (ElapsedMs(start) < max_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
      client.ProcessKeyEvent(weasel::KeyEvent(0));
      const std::string response = Fetch(client);
      ++polls;
      const std::string signature = CandidateSignature(response);
      if (!signature.empty() && signature != baseline_signature) {
        seen_at = ElapsedMs(start);
        break;
      }
      if (HasLmMark(response) && !baseline_marked) {
        seen_at = ElapsedMs(start);
        break;
      }
      if (!LmPending(response))
        break;  // the TSF stops polling here
    }
    client.FocusOut();
    client.EndSession();

    if (seen_at >= 0) {
      visible_ms.push_back(seen_at);
      poll_counts.push_back(polls);
      std::printf("run %d: schema=%s visible=%lld ms after %d polls\n",
                  run + 1, schema_seen.c_str(), seen_at, polls);
    } else {
      std::printf("run %d: schema=%s no LM result within %d ms (%d polls)\n",
                  run + 1, schema_seen.c_str(), max_ms, polls);
    }
  }

  if (visible_ms.empty()) {
    std::printf("RESULT latency keys=%s schema=%s poll=%dms visible=- (none)\n",
                keys, schema_seen.c_str(), poll_ms);
    return 3;
  }
  std::sort(visible_ms.begin(), visible_ms.end());
  std::sort(poll_counts.begin(), poll_counts.end());
  std::printf(
      "RESULT latency keys=%s schema=%s poll=%dms runs=%zu min=%lld median=%lld "
      "max=%lld polls_median=%d\n",
      keys, schema_seen.c_str(), poll_ms, visible_ms.size(), visible_ms.front(),
      visible_ms[visible_ms.size() / 2], visible_ms.back(),
      poll_counts[poll_counts.size() / 2]);
  return 0;
}
