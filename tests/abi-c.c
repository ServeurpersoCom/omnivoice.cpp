/* tests/abi-c.c: link-only ABI smoke test for omnivoice.h.
 *
 * Compiled in pure C99 with -Wall -Werror -pedantic. The purpose of this
 * test is NOT to run a full synthesis (no GGUF loaded, no model required);
 * it is to guarantee at every build that :
 *
 *   1. omnivoice.h parses with a C compiler (no <cstdio>, no std::*, no
 *      C++-only forward declarations leak in).
 *   2. Every public ov_* symbol has C linkage and links from a C
 *      translation unit.
 *   3. The structs are POD and zero-initialisable with `{0}` from C.
 *   4. The ov_log_set callback routes formatted messages from the lib to
 *      the user, and abi_version validation rejects future structs.
 *
 * If this test stops compiling or stops linking, the public ABI has
 * regressed and the build breaks before anything else.
 */

#include "omnivoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool stub_cancel(void * ud) {
    (void) ud;
    return false;
}

/* Counter incremented by the stub log callback. The probe checks that at
 * least one log line was routed through the callback by triggering an
 * ov_init failure (which emits a [OmniVoice] ERROR line via ov_log). */
static int          g_log_lines    = 0;
static enum ov_log_level g_last_log_level = OV_LOG_DEBUG;
static char         g_last_log_msg[512] = { 0 };

static void stub_log(enum ov_log_level level, const char * msg, void * user_data) {
    (void) user_data;
    g_log_lines++;
    g_last_log_level = level;
    if (msg) {
        size_t n = strlen(msg);
        if (n >= sizeof(g_last_log_msg)) {
            n = sizeof(g_last_log_msg) - 1;
        }
        memcpy(g_last_log_msg, msg, n);
        g_last_log_msg[n] = '\0';
    }
}

int main(void) {
    /* Static version string, always reachable. */
    const char * version = ov_version();
    printf("[Probe] %s\n", version);

    /* Default-initialise the public structs from C. */
    struct ov_init_params iparams;
    ov_init_default_params(&iparams);

    struct ov_tts_params params;
    ov_tts_default_params(&params);

    /* Sanity-check a few default values, including the new abi_version. */
    if (params.mg_num_step != 32 || params.chunk_duration_sec <= 0.0f) {
        fprintf(stderr, "[Probe] default values do not match\n");
        return 1;
    }
    if (iparams.abi_version != OV_ABI_VERSION || params.abi_version != OV_ABI_VERSION) {
        fprintf(stderr, "[Probe] abi_version not set by ov_*_default_params\n");
        return 1;
    }

    /* Touch every reference-pointer field, every callback typedef and
     * every output struct field so the compiler validates the layout
     * end-to-end without ever needing a model. */
    params.cancel           = stub_cancel;
    params.cancel_user_data = NULL;

    struct ov_audio audio = { 0 };
    ov_audio_free(&audio);

    struct ov_voice_ref voice_ref = { 0 };
    ov_voice_ref_free(&voice_ref);

    /* Install the log callback before the failing init so the [OmniVoice]
     * ERROR line lands on stub_log instead of stderr. */
    ov_log_set(stub_log, NULL);

    /* Call every entry through its early-return path. ov_init returns
     * NULL on missing model_path, ov_synthesize / ov_duration_sec_to_tokens
     * fail on NULL handle, ov_free is safe on NULL. None of these load a
     * model, but the linker must resolve every name to satisfy the call. */
    struct ov_context * dummy = ov_init(NULL);
    if (dummy != NULL) {
        fprintf(stderr, "[Probe] ov_init(NULL) was supposed to return NULL\n");
        ov_free(dummy);
        return 2;
    }

    /* ov_init(NULL) just failed -> ov_last_error() must point to a
     * non-empty thread-local string. Pointer is always valid (c_str on
     * an empty std::string still gives a NUL byte), so we only need to
     * check the first byte to confirm an error was actually recorded. */
    const char * err = ov_last_error();
    if (err == NULL || err[0] == '\0') {
        fprintf(stderr, "[Probe] ov_last_error() empty after a known failure\n");
        return 5;
    }

    /* The same failure must have surfaced through the log callback at
     * ERROR level. */
    if (g_log_lines == 0) {
        fprintf(stderr, "[Probe] ov_log_set callback never invoked\n");
        return 6;
    }
    if (g_last_log_level != OV_LOG_ERROR) {
        fprintf(stderr, "[Probe] last log level was %d, expected %d\n", (int) g_last_log_level,
                (int) OV_LOG_ERROR);
        return 7;
    }
    printf("[Probe] ov_log_set routed %d line(s), last: '%s'\n", g_log_lines, g_last_log_msg);
    printf("[Probe] ov_last_error reads '%s'\n", err);

    /* abi_version validation : a struct claiming a future ABI must be
     * rejected up front, before any allocation. */
    struct ov_init_params future_iparams;
    ov_init_default_params(&future_iparams);
    future_iparams.model_path  = "irrelevant.gguf";
    future_iparams.abi_version = OV_ABI_VERSION + 1;
    struct ov_context * rejected = ov_init(&future_iparams);
    if (rejected != NULL) {
        fprintf(stderr, "[Probe] ov_init accepted a future abi_version\n");
        ov_free(rejected);
        return 8;
    }

    enum ov_status rc = ov_synthesize(NULL, &params, &audio);
    if (rc != OV_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "[Probe] ov_synthesize(NULL) returned %d, expected %d\n", (int) rc,
                (int) OV_STATUS_INVALID_PARAMS);
        return 3;
    }

    int frames = ov_duration_sec_to_tokens(NULL, 1.0f);
    if (frames < 1) {
        fprintf(stderr, "[Probe] ov_duration_sec_to_tokens returned %d, expected >= 1\n", frames);
        return 4;
    }

    enum ov_status vrc = ov_extract_voice_ref(NULL, NULL, 0, &voice_ref);
    if (vrc != OV_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "[Probe] ov_extract_voice_ref(NULL) returned %d, expected %d\n", (int) vrc,
                (int) OV_STATUS_INVALID_PARAMS);
        return 9;
    }

    /* Restore the default stderr fallback before exit so the trailing
     * [OmniVoice] log lines from the cleanup paths land where the user
     * expects them. */
    ov_log_set(NULL, NULL);

    ov_free(NULL);
    ov_audio_free(&audio);
    ov_voice_ref_free(&voice_ref);

    return 0;
}
