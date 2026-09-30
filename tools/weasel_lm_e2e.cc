// Headless end-to-end test for the Weasel + librime + lm_ranker async path.
//
// This is intentionally a separate process.  It uses the production rime.dll
// and the same Rime API notification contract as WeaselServer, but it neither
// registers a TSF text service nor changes the installed input method.
//
// Example:
//   weasel_lm_e2e.exe --shared data --user user --context "已有中文上文"
//       --keys zheshiyigeceshi --expect-lm

#include <rime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace {

std::string Option(int argc, char** argv, const char* name,
                   const char* fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0)
      return argv[i + 1];
  }
  return fallback;
}

bool HasFlag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0)
      return true;
  }
  return false;
}

struct Host {
  std::mutex mutex;
  std::condition_variable cv;
  unsigned notifications = 0;

  static void OnNotify(void* object, uintptr_t, const char* type,
                       const char*) {
    if (!object || !type || std::strcmp(type, "lm_ranker") != 0)
      return;
    auto* host = static_cast<Host*>(object);
    {
      std::lock_guard<std::mutex> lock(host->mutex);
      ++host->notifications;
    }
    host->cv.notify_one();
  }

  bool WaitForNotification(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    if (notifications)
      return true;
    return cv.wait_for(lock, timeout, [this] { return notifications != 0; });
  }

  bool ConsumeNotification() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!notifications)
      return false;
    --notifications;
    return true;
  }
};

bool DumpCandidates(RimeApi* rime, RimeSessionId session, bool* has_lm) {
  RIME_STRUCT(RimeContext, context);
  if (!rime->get_context(session, &context))
    return false;
  std::printf("candidates=%d composing=%d\n", context.menu.num_candidates,
              context.composition.length > 0 ? 1 : 0);
  for (int i = 0; i < context.menu.num_candidates; ++i) {
    const RimeCandidate& candidate = context.menu.candidates[i];
    const char* text = candidate.text ? candidate.text : "";
    const char* comment = candidate.comment ? candidate.comment : "";
    const char* type = candidate.type ? candidate.type : "";
    std::printf("  %d\t%s\t[%s]\t%s\n", i + 1, text, type, comment);
    if (std::strstr(comment, "[LM]") || std::strcmp(type, "lm_ranker") == 0)
      *has_lm = true;
  }
  rime->free_context(&context);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string shared = Option(argc, argv, "--shared");
  const std::string user = Option(argc, argv, "--user");
  const std::string schema = Option(argc, argv, "--schema", "luna_pinyin_simp");
  const std::string context = Option(argc, argv, "--context");
  const std::string keys = Option(argc, argv, "--keys", "zheshiyigeceshi");
  const int timeout_ms = std::atoi(Option(argc, argv, "--timeout-ms", "8000").c_str());
  const bool expect_lm = HasFlag(argc, argv, "--expect-lm");
  if (shared.empty() || user.empty() || context.empty() || keys.empty()) {
    std::fprintf(stderr,
                 "usage: %s --shared DIR --user DIR --context UTF-8 --keys "
                 "PINYIN [--schema ID] [--timeout-ms N] [--expect-lm]\n",
                 argv[0]);
    return 2;
  }

  RimeApi* rime = rime_get_api();
  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = shared.c_str();
  traits.prebuilt_data_dir = shared.c_str();
  traits.user_data_dir = user.c_str();
  traits.distribution_name = "Weasel LM E2E";
  traits.distribution_code_name = "weasel";
  traits.distribution_version = "test";
  traits.app_name = "rime.weasel-lm-e2e";
  rime->setup(&traits);

  Host host;
  // This ordering intentionally matches the repaired WeaselServer: the
  // callback is installed after RimeInitialize, then refresh work runs on the
  // host's serialized event loop below.
  rime->initialize(&traits);
  if (rime->start_maintenance(True))
    rime->join_maintenance_thread();
  rime->set_notification_handler(&Host::OnNotify, &host);
  RimeSessionId session = rime->create_session();
  if (!session || !rime->select_schema(session, schema.c_str())) {
    std::fprintf(stderr, "cannot create/select schema '%s'\n", schema.c_str());
    rime->finalize();
    return 1;
  }
  rime->set_property(session, "surrounding_text", context.c_str());
  if (!rime->simulate_key_sequence(session, keys.c_str())) {
    std::fprintf(stderr, "simulate_key_sequence failed for '%s'\n", keys.c_str());
    rime->destroy_session(session);
    rime->finalize();
    return 1;
  }

  bool has_lm = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    // Candidate retrieval drives Rime's lazy filter pipeline, just as a
    // Weasel UI update does.
    DumpCandidates(rime, session, &has_lm);
    if (has_lm)
      break;
    const auto now = std::chrono::steady_clock::now();
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    host.WaitForNotification(std::min(remaining, std::chrono::milliseconds(100)));
    while (host.ConsumeNotification()) {
      if (RIME_API_AVAILABLE(rime, refresh_non_confirmed_composition)) {
        const Bool refreshed = rime->refresh_non_confirmed_composition(session);
        std::printf("lm notification -> refresh_non_confirmed=%d\n",
                    refreshed ? 1 : 0);
      }
    }
  }

  std::printf("RESULT lm_candidate=%s\n", has_lm ? "yes" : "no");
  rime->destroy_session(session);
  rime->finalize();
  return expect_lm && !has_lm ? 3 : 0;
}
