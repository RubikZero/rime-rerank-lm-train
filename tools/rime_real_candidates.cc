// Extract real IME candidates from librime for pinyin key sequences.
// Input TSV:  id \t keys \t target
// Output TSV: id \t rank \t cand1 \x1e comment1 \x1f cand2 \x1e comment2 ...
//   rank = 0-based position of target in the (page-flattened) candidate list,
//   -1 if not found. Candidates are capped at --max.

#include <rime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static std::string opt(int argc, char** argv, const char* name,
                       const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      return argv[i + 1];
    }
  }
  return fallback;
}

int main(int argc, char** argv) {
  std::string shared = opt(argc, argv, "--shared", "");
  std::string user = opt(argc, argv, "--user", "");
  std::string schema = opt(argc, argv, "--schema", "luna_pinyin_simp");
  std::string input = opt(argc, argv, "--input", "");
  std::string output = opt(argc, argv, "--output", "");
  std::string meta_path = opt(argc, argv, "--meta", "");
  int wait_ms = std::atoi(opt(argc, argv, "--wait", "0").c_str());
  std::string parse_out = opt(argc, argv, "--parse-out", "");
  FILE* fparse = parse_out.empty() ? nullptr : std::fopen(parse_out.c_str(), "wb");
  int max_candidates = std::atoi(opt(argc, argv, "--max", "27").c_str());
  int max_pages = std::atoi(opt(argc, argv, "--pages", "6").c_str());

  if (shared.empty() || user.empty() || input.empty() || output.empty()) {
    std::fprintf(stderr,
                 "usage: %s --shared DIR --user DIR [--schema ID] "
                 "--input TSV --output TSV [--max 27]\n",
                 argv[0]);
    return 2;
  }

  FILE* fin = std::fopen(input.c_str(), "rb");
  FILE* fout = std::fopen(output.c_str(), "wb");
  if (!fin || !fout) {
    std::fprintf(stderr, "cannot open input/output\n");
    return 2;
  }

  RimeApi* rime = rime_get_api();
  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = shared.c_str();
  traits.user_data_dir = user.c_str();
  traits.app_name = "rime.realcand";
  traits.distribution_name = "Weasel";
  traits.distribution_code_name = "weasel";
  traits.distribution_version = "0.17.4";

  rime->setup(&traits);
  rime->initialize(&traits);
  if (rime->start_maintenance(True)) {
    rime->join_maintenance_thread();
  }

  RimeSessionId session = rime->create_session();
  if (!session) {
    std::fprintf(stderr, "failed to create session\n");
    return 1;
  }
  if (!rime->select_schema(session, schema.c_str())) {
    std::fprintf(stderr, "failed to select schema %s\n", schema.c_str());
  }

  char line[16384];
  std::map<std::string, std::string> contexts;
  if (!meta_path.empty()) {
    FILE* fm = std::fopen(meta_path.c_str(), "rb");
    if (fm) {
      char mline[65536];
      while (std::fgets(mline, sizeof mline, fm)) {
        std::string row(mline);
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r'))
          row.pop_back();
        size_t t1 = row.find('\t');
        size_t t2 = t1 == std::string::npos ? std::string::npos
                                            : row.find('\t', t1 + 1);
        size_t t3 = t2 == std::string::npos ? std::string::npos
                                            : row.find('\t', t2 + 1);
        if (t3 == std::string::npos)
          continue;
        contexts[row.substr(0, t1)] = row.substr(t3 + 1);
      }
      std::fclose(fm);
    }
    std::fprintf(stderr, "contexts loaded: %zu\n", contexts.size());
  }
  long total = 0, found = 0;
  while (std::fgets(line, sizeof line, fin)) {
    std::string row(line);
    while (!row.empty() && (row.back() == '\n' || row.back() == '\r')) {
      row.pop_back();
    }
    if (row.empty()) {
      continue;
    }
    size_t t1 = row.find('\t');
    size_t t2 = t1 == std::string::npos ? std::string::npos
                                        : row.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) {
      continue;
    }
    std::string id = row.substr(0, t1);
    std::string keys = row.substr(t1 + 1, t2 - t1 - 1);
    std::string target = row.substr(t2 + 1);

    rime->clear_composition(session);
    auto ctx_it = contexts.find(id);
    if (ctx_it != contexts.end()) {
      rime->set_property(session, "surrounding_text", ctx_it->second.c_str());
    }
    std::vector<std::string> cands, comments;
    std::vector<double> qualities;
    std::vector<std::string> types;
    int rank = -1;
    if (!keys.empty() && rime->simulate_key_sequence(session, keys.c_str())) {
      if (wait_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        if (RIME_API_AVAILABLE(rime, refresh_non_confirmed_composition)) {
          Bool refreshed = rime->refresh_non_confirmed_composition(session);
          std::fprintf(stderr, "id=%s refresh_non_confirmed=%d\n", id.c_str(),
                       refreshed);
          std::fflush(stderr);
        }
      }
      if (fparse && RIME_API_AVAILABLE(rime, get_segmentation)) {
        char syl[1024] = {0};
        char words[512] = {0};
        char word_texts[1024] = {0};
        Bool ok0 = rime->get_segmentation(session, 0, syl, sizeof syl);
        Bool ok1 = rime->get_segmentation(session, 1, words, sizeof words);
        rime->get_segmentation(session, 2, word_texts, sizeof word_texts);
        std::fprintf(fparse, "%s\t%s\t%s\t%s\t%d\t%d\n", id.c_str(), syl,
                     words, word_texts, ok0 ? 1 : 0, ok1 ? 1 : 0);
      }
      for (int page = 0; page < max_pages &&
                         (int)cands.size() < max_candidates;
           ++page) {
        RIME_STRUCT(RimeContext, ctx);
        if (!rime->get_context(session, &ctx)) {
          break;
        }
        int n = ctx.menu.num_candidates;
        for (int i = 0; i < n && (int)cands.size() < max_candidates; ++i) {
          const char* text = ctx.menu.candidates[i].text;
          if (!text) {
            continue;
          }
          if (rank < 0 && target == text) {
            rank = (int)cands.size();
          }
          cands.push_back(text);
          const char* comment = ctx.menu.candidates[i].comment;
          comments.push_back(comment ? comment : "");
          qualities.push_back(ctx.menu.candidates[i].quality);
          const char* type = ctx.menu.candidates[i].type;
          types.push_back(type ? type : "");
        }
        bool last = ctx.menu.is_last_page;
        rime->free_context(&ctx);
        if (last) {
          break;
        }
        if (!rime->change_page(session, False)) {
          break;
        }
      }
    }
    ++total;
    if (rank >= 0) {
      ++found;
    }
    std::fprintf(fout, "%s\t%d\t", id.c_str(), rank);
    for (size_t i = 0; i < cands.size(); ++i) {
      if (i) {
        std::fputc(0x1f, fout);
      }
      std::fputs(cands[i].c_str(), fout);
      std::fputc(0x1e, fout);
      std::fputs(comments[i].c_str(), fout);
      std::fputc(0x1e, fout);
      std::fprintf(fout, "%.9g", qualities[i]);
      std::fputc(0x1e, fout);
      std::fputs(types[i].c_str(), fout);
    }
    std::fputc('\n', fout);
    if (total % 100 == 0) {
      std::fprintf(stderr, "processed %ld, found %ld\n", total, found);
      std::fflush(stderr);
    }
  }

  rime->destroy_session(session);
  rime->finalize();
  std::fclose(fin);
  std::fclose(fout);
  if (fparse)
    std::fclose(fparse);
  std::fprintf(stderr, "done: total=%ld found=%ld (%.1f%%)\n", total, found,
               total ? 100.0 * found / total : 0.0);
  return 0;
}
