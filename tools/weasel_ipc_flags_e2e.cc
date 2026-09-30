// Verifies the client-side classification the input-method defence relies on:
// "the server answered: this key is not handled" must be distinguishable from
// "the server never answered".
//
// Phase 1 runs against a healthy isolated server (transaction must succeed,
// even though a zero key code is not handled by Rime).  Then the caller stalls
// the server (thread suspension is done by the test script during the pause
// below) and phase 2 must report a *failed transaction* while still returning
// within its budget instead of hanging.
//
// Usage: weasel_ipc_flags_e2e.exe [pause_before_phase2_ms]

#include <WeaselIPC.h>
#include <PipeChannel.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

std::wstring ToWide(const char* value) {
  const int length = MultiByteToWideChar(CP_UTF8, 0, value, -1, NULL, 0);
  std::wstring result(length > 0 ? length - 1 : 0, L'\0');
  if (length > 0)
    MultiByteToWideChar(CP_UTF8, 0, value, -1, result.data(), length);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const int pause_ms = argc > 1 ? std::atoi(argv[1]) : 0;
  std::printf("per-transaction budget: %lu ms\n",
              static_cast<unsigned long>(weasel::IpcReadTimeoutMs()));

  weasel::Client client;
  if (!client.Connect()) {
    std::fprintf(stderr, "cannot connect to isolated WeaselServer\n");
    return 2;
  }
  client.StartSession();
  if (!client.Echo()) {
    std::fprintf(stderr, "cannot start an isolated Weasel session\n");
    return 2;
  }
  client.FocusIn();
  client.SetSurroundingText(ToWide("meeting at three pm, "));

  // Phase 1: healthy server.
  bool failed = true;
  const bool handled =
      client.ProcessKeyEventChecked(weasel::KeyEvent(0, 0), &failed);
  std::printf("phase1 healthy: handled=%d transaction_failed=%d\n",
              handled ? 1 : 0, failed ? 1 : 0);
  const bool phase1_ok = !failed;

  if (pause_ms > 0) {
    std::printf("waiting %d ms for the server to be stalled...\n", pause_ms);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
  }

  // Phase 2: the caller has stalled the server in the meantime.
  const auto start = std::chrono::steady_clock::now();
  bool failed2 = false;
  client.ProcessKeyEventChecked(weasel::KeyEvent('a', 0), &failed2);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  std::printf("phase2 stalled: transaction_failed=%d after %lld ms\n",
              failed2 ? 1 : 0, static_cast<long long>(elapsed));

  const bool pass = phase1_ok && failed2 && elapsed < 20000;
  std::printf("RESULT flags_e2e=pass healthy_ok=%d stalled_detected=%d\n",
              phase1_ok ? 1 : 0, failed2 ? 1 : 0);
  return pass ? 0 : 3;
}
