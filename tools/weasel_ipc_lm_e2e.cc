// Drives a real WeaselServer through its public IPC protocol.  Set
// RIME_WEASEL_PIPE_NAME to the private pipe used by the isolated server.

#include <WeaselIPC.h>

#include <boost/interprocess/streams/bufferstream.hpp>

#include <chrono>
#include <cstdlib>
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

std::wstring ToWide(const char* value) {
  const int length = MultiByteToWideChar(CP_UTF8, 0, value, -1, NULL, 0);
  std::wstring result(length, L'\0');
  if (length > 0) {
    MultiByteToWideChar(CP_UTF8, 0, value, -1, result.data(), length);
    result.resize(length - 1);
  }
  return result;
}

bool PrintResponse(weasel::Client& client, bool* has_lm) {
  WCHAR buffer[WEASEL_IPC_BUFFER_LENGTH] = {0};
  if (!client.GetResponseData(
          std::bind(ReadResponse, std::placeholders::_1, std::placeholders::_2,
                    buffer)))
    return false;
  const std::string response = ToUtf8(buffer);
  std::printf("%s\n", response.c_str());
  if (response.find("[LM]") != std::string::npos ||
      response.find("candidate.type=lm_ranker") != std::string::npos)
    *has_lm = true;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* context = argc > 1 ? argv[1] : "会议将在下午三点举行，";
  const char* keys = argc > 2 ? argv[2] : "zheshiyigeceshi";
  const int timeout_ms = argc > 3 ? std::atoi(argv[3]) : 10000;
  // Typing speed matters: an interactive user leaves the language model time to
  // answer between keys, while a zero delay fires every key before the model is
  // even loaded (which hides the rerank entirely).
  const int key_delay_ms = argc > 4 ? std::atoi(argv[4]) : 0;
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
  bool has_lm = false;
  client.FocusIn();
  const std::wstring context_wide = ToWide(context);
  client.SetSurroundingText(context_wide);
  for (const char* key = keys; *key; ++key) {
    client.ProcessKeyEvent(weasel::KeyEvent(*key, 0));
    PrintResponse(client, &has_lm);
    if (key_delay_ms > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(key_delay_ms));
  }

  // The server owns the asynchronous refresh.  A harmless Shift probe after
  // the wait asks it to serialize the current candidate state over IPC, so
  // this console program can assert what its UI would display.
  for (int elapsed = 0; elapsed < timeout_ms && !has_lm; elapsed += 250) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    client.ProcessKeyEvent(weasel::KeyEvent(VK_SHIFT, 0));
    PrintResponse(client, &has_lm);
  }
  client.FocusOut();
  client.EndSession();
  std::printf("RESULT lm_candidate=%s\n", has_lm ? "yes" : "no");
  return has_lm ? 0 : 3;
}
