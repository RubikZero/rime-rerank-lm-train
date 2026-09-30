// Measures the user-visible latency of the asynchronous LM rerank: how long
// after the last keystroke the candidate menu actually shows the LM result.
//
// It drives an isolated WeaselServer exactly like the TSF client does:
//   keys with a realistic gap -> poll every `poll_ms` with ProcessKeyEvent(0)
//   and read the response -> stop at the first response whose candidate list
//   differs from the IME baseline (or that carries [LM]).
//
// Usage: weasel_ipc_latency_e2e.exe <context> <keys> <key_delay_ms> <poll_ms> [max_ms] [runs]
// Output: one RESULT line per run with the measured latency and how it was
// detected, so a script can aggregate min/median/max.

#include <WeaselIPC.h>

#include <boost/interprocess/streams/bufferstream.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

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
  std::wstring result(length, L'\0');
  if (length > 0) {
    MultiByteToWideChar(CP_UTF8, 0, value, -1, result.data(), length);
    result.resize(length - 1);
  }
  return result;
}

std::string Fetch(weasel::Client& client) {
  WCHAR buffer[WEASEL_IPC_BUFFER_LENGTH] = {0};
  if (!client.GetResponseData(
          std::bind(ReadResponse, std::placeholders::_1, std::placeholders::_2,
                    buffer)))
    return {};
  return ToUtf8(buffer);
}

// The candidate list as it appears in the response: "ctx.cand=" carries a
// serialized CandidateInfo, whose candidate texts are the visible order.
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

// config.lm_pending=1 while the plugin still has rerank/sentence work for this
// composition.  The TSF client cancels its poll chain when it sees 0, so the
// probe reports how many transactions that costs and whether the menu had
// already been refreshed by then (the invariant that makes the early stop safe).
bool LmPending(const std::string& response) {
  const std::string key = "config.lm_pending=";
  const size_t at = response.find(key);
  if (at == std::string::npos)
    return true;  // unknown: keep polling
  return response.compare(at + key.size(), 1, "1") == 0;
}

long long ElapsedMs(const std::chrono::steady_clock::time_point& from) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - from)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  const char* context = argc > 1 ? argv[1] : "meeting at three pm, ";
  const char* keys = argc > 2 ? argv[2] : "beijing";
  const int key_delay_ms = argc > 3 ? std::atoi(argv[3]) : 120;
  const int poll_ms = argc > 4 ? std::atoi(argv[4]) : 120;
  const int max_ms = argc > 5 ? std::atoi(argv[5]) : 3000;
  const int runs = argc > 6 ? std::atoi(argv[6]) : 5;

  weasel::Client client;
  if (!client.Connect()) {
    std::fprintf(stderr, "cannot connect to isolated WeaselServer\n");
    return 2;
  }

  std::vector<long long> visible_ms;
  std::vector<int> poll_counts;
  int detections_by_order = 0;
  int detections_by_mark = 0;
  int warmup_failed = 0;
  int unsafe_early_stop = 0;

  for (int run = 0; run < runs; ++run) {
    client.StartSession();
    if (!client.Echo()) {
      // The very first session may race with model loading; count it and retry.
      ++warmup_failed;
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      --run;
      if (warmup_failed > 6)
        return 2;
      continue;
    }
    client.FocusIn();
    // A unique context per run: the rerank cache is keyed by the full
    // (context, candidate, quality) snapshot, so repeating the same context
    // would let a later run start from an already ranked menu and the
    // "menu changed" signal could never fire.
    const std::string run_context =
        std::string(context) + " run" + std::to_string(run);
    client.SetSurroundingText(ToWide(run_context.c_str()));

    // Type the input, then measure from the last keystroke.
    std::string baseline;
    for (const char* key = keys; *key; ++key) {
      client.ProcessKeyEvent(weasel::KeyEvent(*key, 0));
      baseline = Fetch(client);
      if (key_delay_ms > 0 && *(key + 1))
        std::this_thread::sleep_for(std::chrono::milliseconds(key_delay_ms));
    }
    const std::string baseline_signature = CandidateSignature(baseline);
    const bool baseline_marked = HasLmMark(baseline);

    const auto start = std::chrono::steady_clock::now();
    long long seen_at = -1;
    int polls = 0;
    int idle_at_poll = -1;
    int changed_at_poll = -1;
    bool changed_before_idle = false;
    bool by_order = false;
    bool by_mark = false;
    while (ElapsedMs(start) < max_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
      client.ProcessKeyEvent(weasel::KeyEvent(0));
      const std::string response = Fetch(client);
      ++polls;
      const std::string signature = CandidateSignature(response);
      const bool marked = HasLmMark(response);
      if (seen_at < 0 && !signature.empty() && signature != baseline_signature) {
        seen_at = ElapsedMs(start);
        changed_at_poll = polls;
        by_order = true;
      }
      if (seen_at < 0 && marked && !baseline_marked) {
        seen_at = ElapsedMs(start);
        changed_at_poll = polls;
        by_mark = true;
      }
      if (!LmPending(response)) {
        idle_at_poll = polls;
        changed_before_idle = seen_at >= 0;
        break;  // the TSF stops polling here
      }
    }
    // The early stop is only safe if the menu had already been refreshed when
    // the server reported "nothing pending".
    if (idle_at_poll > 0 && !changed_before_idle)
      ++unsafe_early_stop;

    client.FocusOut();
    client.EndSession();

    if (seen_at >= 0) {
      visible_ms.push_back(seen_at);
      poll_counts.push_back(polls);
      if (by_order)
        ++detections_by_order;
      if (by_mark)
        ++detections_by_mark;
      std::printf(
          "run %d: visible=%lld ms after %d polls (%s), idle at poll %d, "
          "menu updated by then=%s\n",
          run + 1, seen_at, polls, by_order ? "order changed" : "mark appeared",
          idle_at_poll, changed_before_idle ? "yes" : "NO");
    } else {
      std::printf(
          "run %d: no LM result within %d ms (%d polls, idle at %d)\n", run + 1,
          max_ms, polls, idle_at_poll);
    }
  }

  if (visible_ms.empty()) {
    std::printf("RESULT latency keys=%s poll=%dms visible=- (none)\n", keys,
                poll_ms);
    return 3;
  }
  std::sort(visible_ms.begin(), visible_ms.end());
  std::sort(poll_counts.begin(), poll_counts.end());
  const long long median = visible_ms[visible_ms.size() / 2];
  std::printf(
      "RESULT latency keys=%s poll=%dms runs=%zu min=%lld median=%lld max=%lld "
      "polls_median=%d by_order=%d by_mark=%d unsafe_early_stop=%d\n",
      keys, poll_ms, visible_ms.size(), visible_ms.front(), median,
      visible_ms.back(), poll_counts[poll_counts.size() / 2],
      detections_by_order, detections_by_mark, unsafe_early_stop);
  return unsafe_early_stop == 0 ? 0 : 4;
}
