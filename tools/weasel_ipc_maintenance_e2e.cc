// Verifies the maintenance round-trip over the public IPC protocol:
//   StartMaintenance -> (service disabled) -> EndMaintenance -> typing works.
//
// Regression: EndMaintenance() used to re-initialize librime synchronously
// inside the request, holding the global api lock for as long as
// join_maintenance_thread() took (a full deployment can be minutes).  Every
// other session timed out during that window, and if the client gave up before
// the resume finished, the service could stay disabled and answer every key
// with FALSE.  The resume now happens on a background thread, so this client
// must get an immediate answer and the server must accept keys again -- within
// the wait below -- without any further client action.
//
// Set RIME_WEASEL_PIPE_NAME / RIME_WEASEL_USER_DIR for an isolated server.

#include <WeaselIPC.h>

#include <boost/interprocess/streams/bufferstream.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

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

bool PollOnce(weasel::Client& client, bool* has_lm) {
  WCHAR buffer[WEASEL_IPC_BUFFER_LENGTH] = {0};
  if (!client.GetResponseData(
          std::bind(ReadResponse, std::placeholders::_1, std::placeholders::_2,
                    buffer)))
    return false;
  const std::string response = ToUtf8(buffer);
  if (response.find("[LM]") != std::string::npos ||
      response.find("candidate.type=lm_ranker") != std::string::npos)
    *has_lm = true;
  return true;
}

// Types a short probe and reports what the server answered.
bool TryTyping(weasel::Client& client, int timeout_ms, bool* has_lm) {
  client.StartSession();
  if (!client.Echo())
    return false;
  client.FocusIn();
  // Same cursor context as the interactive path: the language model consumes
  // session.surrounding_text, and without it the rerank has no context.
  const char* context = "会议将在下午三点举行，";
  const int wide_len = MultiByteToWideChar(CP_UTF8, 0, context, -1, NULL, 0);
  std::wstring context_wide(wide_len > 0 ? wide_len - 1 : 0, L'\0');
  if (wide_len > 0)
    MultiByteToWideChar(CP_UTF8, 0, context, -1, context_wide.data(), wide_len);
  client.SetSurroundingText(context_wide);
  for (const char* key = "beijingdaxue"; *key; ++key) {
    client.ProcessKeyEvent(weasel::KeyEvent(*key, 0));
    PollOnce(client, has_lm);
  }
  for (int elapsed = 0; elapsed < timeout_ms && !*has_lm; elapsed += 250) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    client.ProcessKeyEvent(weasel::KeyEvent(VK_SHIFT, 0));
    PollOnce(client, has_lm);
  }
  client.FocusOut();
  client.EndSession();
  return *has_lm;
}

}  // namespace

int main(int argc, char** argv) {
  const int max_wait_ms = argc > 1 ? std::atoi(argv[1]) : 30000;
  weasel::Client client;
  if (!client.Connect()) {
    std::fprintf(stderr, "cannot connect to isolated WeaselServer\n");
    return 2;
  }

  bool has_lm = false;
  if (!TryTyping(client, 10000, &has_lm)) {
    std::printf("RESULT warmup=no\n");
    return 3;
  }
  std::printf("warmup ok (lm=%d)\n", has_lm ? 1 : 0);

  // Enter maintenance mode, as the deployer does before deploying.
  auto watch = std::chrono::steady_clock::now();
  client.StartMaintenance();
  const auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - watch)
                            .count();
  std::printf("StartMaintenance answered in %lld ms\n",
              static_cast<long long>(start_ms));

  // While disabled, keys must be refused quickly rather than hang.
  watch = std::chrono::steady_clock::now();
  client.StartSession();
  const bool disabled_session = client.Echo();
  client.ProcessKeyEvent(weasel::KeyEvent('a', 0));
  const auto disabled_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - watch)
                               .count();
  std::printf("disabled session=%d answered in %lld ms\n",
              disabled_session ? 1 : 0,
              static_cast<long long>(disabled_ms));

  // Resume: this must come back immediately, with the re-initialization
  // running in the background.
  watch = std::chrono::steady_clock::now();
  client.EndMaintenance();
  const auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - watch)
                          .count();
  std::printf("EndMaintenance answered in %lld ms\n",
              static_cast<long long>(end_ms));

  // The service must accept input again without any further client action.
  bool recovered = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(max_wait_ms);
  int attempt = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    bool lm = false;
    ++attempt;
    if (TryTyping(client, 2000, &lm)) {
      recovered = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  const auto recover_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - watch)
                              .count();
  std::printf("typing recovered=%d after %lld ms (%d attempts)\n",
              recovered ? 1 : 0, static_cast<long long>(recover_ms), attempt);

  const bool fast_end = end_ms < 1500;
  const bool pass = recovered && fast_end;
  std::printf("RESULT maintenance_end_ms=%lld recovered=%d pass=%d\n",
              static_cast<long long>(end_ms), recovered ? 1 : 0, pass ? 1 : 0);
  return pass ? 0 : 4;
}
