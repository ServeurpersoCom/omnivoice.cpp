#pragma once
// text-chunker-stream.h: incremental wrapper around chunk_text_punctuation.
//
// Equivalence guarantee: at flush_eof, the concatenation in order of every
// chunk emitted by push_bytes + flush_eof is identical (codepoint by
// codepoint) to chunk_text_punctuation(full_text, chunk_len,
// OMNIVOICE_MIN_CHUNK_LEN) called once on the fully accumulated input.
// Same chunks, same boundaries, same fold decisions. The streaming path
// trades a one chunk look ahead delay for this exact equivalence: a chunk
// that just closed is held back until the next one is observed, so the
// min_chunk_len fold rule can fire:
//   - first chunk shorter than min_chunk_len folds into the second one
//   - chunk N > 0 shorter than min_chunk_len folds into chunk N - 1
// The look ahead delay is bounded by one chunk, dominated in practice by
// the time the upstream LLM takes to produce ~chunk_len codepoints, which
// is far below the synthesis time on a single chunk on GPU. So the
// streaming path emits the same exact text chunks as the buffered path,
// just one chunk later.

#include "text-chunker.h"

#include <string>
#include <vector>

struct text_chunker_stream {
    std::string buffer;
    int         chunk_len;
    int         min_chunk_len;
    int         n_seen;   // number of chunks observed in the offline
                          // re-parse so far; index of the next chunk
                          // to enter the look ahead pipeline.
    bool        has_pending;
    std::string pending;  // chunk waiting in the look ahead slot.

    void init(int chunk_len_, int min_chunk_len_) {
        buffer.clear();
        chunk_len     = chunk_len_;
        min_chunk_len = min_chunk_len_;
        n_seen        = 0;
        has_pending   = false;
        pending.clear();
    }

    // Append bytes, rerun the offline chunker on the full buffer, advance
    // the look ahead pipeline. Emits chunks that are no longer at risk of
    // a fold. The very last chunk in the offline result is always kept
    // back since a future sentence might still extend it; the second to
    // last is also kept back so the fold rule has a chance to fire.
    std::vector<std::string> push_bytes(const char * data, size_t n) {
        if (n > 0) {
            buffer.append(data, n);
        }
        std::vector<std::string> all = chunk_text_punctuation(buffer, chunk_len, 0);
        return advance(all, false);
    }

    // EOF drain: every chunk in the offline result is now stable. The
    // look ahead pipeline runs to completion and the chunker comes out
    // fresh, so a caller can keep pushing a new stream after the drain
    // (line oriented streaming flushes at every newline).
    std::vector<std::string> flush_eof() {
        std::vector<std::string> all = chunk_text_punctuation(buffer, chunk_len, 0);
        std::vector<std::string> out = advance(all, true);
        buffer.clear();
        n_seen = 0;
        return out;
    }

  private:
    // Pump every newly observed chunk (index >= n_seen) through the look
    // ahead slot. An incoming chunk either folds into pending (fold rule
    // fires) or pushes the previous pending out (which is then either
    // emitted or folded forward when applicable). At EOF the trailing
    // pending is emitted as is.
    std::vector<std::string> advance(const std::vector<std::string> & all, bool eof) {
        std::vector<std::string> out;

        // The last chunk of the offline result is open: a future sentence
        // could still extend it. Stop one before, except at EOF where the
        // last one is also stable.
        int n_stable = (int) all.size();
        if (!eof) {
            n_stable -= 1;
        }
        if (n_stable < 0) {
            n_stable = 0;
        }

        for (int i = n_seen; i < n_stable; i++) {
            const std::string & incoming = all[i];

            if (!has_pending) {
                pending     = incoming;
                has_pending = true;
                continue;
            }

            // Fold rule. The first chunk, if short, folds into the second
            // one (pending becomes pending + incoming, kept in the slot).
            // Any later short chunk folds into the previous one (pending
            // becomes pending + incoming, still kept in the slot).
            int incoming_cp = chunker_utf8_count(incoming);
            int pending_cp  = chunker_utf8_count(pending);

            bool incoming_short          = (incoming_cp < min_chunk_len);
            bool pending_first_and_short = (i == 1) && (pending_cp < min_chunk_len);

            if (incoming_short) {
                // Fold incoming into pending. pending stays in slot.
                pending += incoming;
                continue;
            }
            if (pending_first_and_short) {
                // Fold the very first short chunk into the incoming one.
                // The combined chunk takes the slot.
                pending = pending + incoming;
                continue;
            }
            // Stable case: emit the pending, the incoming takes the slot.
            out.push_back(pending);
            pending = incoming;
        }
        n_seen = n_stable;

        if (eof && has_pending) {
            out.push_back(pending);
            pending.clear();
            has_pending = false;
        }

        return out;
    }
};
