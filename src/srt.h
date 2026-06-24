#pragma once
// srt.h: minimal SubRip (.srt) parser for the dub timeline.
//
// Parses an SRT byte buffer into a list of cues, each carrying a start and
// end time in seconds plus the joined text. Tolerant of CRLF, a leading
// UTF-8 BOM, a missing index line, comma or period as the millisecond
// separator, and multi line cue text (joined with a single space). This is
// a CLI side utility like audio-io.h, never part of the public ABI.

#include <string>
#include <vector>

struct SrtCue {
    int         index;  // 1-based source index, 0 when absent
    double      t0;     // start time in seconds
    double      t1;     // end time in seconds
    std::string text;   // joined cue text
};

// Read a run of decimal digits starting at i. Returns false when no digit
// is present, otherwise advances i past the run and stores the value.
static bool srt_read_uint(const std::string & s, size_t & i, int & out) {
    size_t start = i;
    long   v     = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + (s[i] - '0');
        i++;
    }
    if (i == start) {
        return false;
    }
    out = (int) v;
    return true;
}

// Parse one "HH:MM:SS,mmm" stamp into seconds. Accepts '.' for the
// millisecond separator and any digit count for the fractional part,
// scaling it to milliseconds. Rejects trailing garbage.
static bool srt_parse_time(std::string t, double & out) {
    size_t a = t.find_first_not_of(" \t\r\n");
    size_t b = t.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return false;
    }
    t = t.substr(a, b - a + 1);

    size_t i = 0;
    int    h = 0, m = 0, s = 0, ms = 0;
    if (!srt_read_uint(t, i, h) || i >= t.size() || t[i] != ':') {
        return false;
    }
    i++;
    if (!srt_read_uint(t, i, m) || i >= t.size() || t[i] != ':') {
        return false;
    }
    i++;
    if (!srt_read_uint(t, i, s) || i >= t.size() || (t[i] != ',' && t[i] != '.')) {
        return false;
    }
    i++;
    size_t ms_start = i;
    if (!srt_read_uint(t, i, ms) || i != t.size()) {
        return false;
    }

    double msd  = (double) ms;
    int    n_ms = (int) (i - ms_start);
    for (int k = n_ms; k < 3; k++) {
        msd *= 10.0;
    }
    for (int k = n_ms; k > 3; k--) {
        msd /= 10.0;
    }

    out = h * 3600.0 + m * 60.0 + s + msd / 1000.0;
    return true;
}

// Split into logical lines on '\n', stripping a trailing '\r' and a leading
// UTF-8 BOM. Drops nothing else so blank lines stay as block separators.
static std::vector<std::string> srt_split_lines(const std::string & raw) {
    size_t p = 0;
    if (raw.size() >= 3 && (unsigned char) raw[0] == 0xEF && (unsigned char) raw[1] == 0xBB &&
        (unsigned char) raw[2] == 0xBF) {
        p = 3;
    }

    std::vector<std::string> lines;
    std::string              cur;
    for (size_t i = p; i < raw.size(); i++) {
        char c = raw[i];
        if (c == '\n') {
            if (!cur.empty() && cur.back() == '\r') {
                cur.pop_back();
            }
            lines.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        if (cur.back() == '\r') {
            cur.pop_back();
        }
        lines.push_back(cur);
    }
    return lines;
}

static bool srt_is_blank(const std::string & s) {
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\r') {
            return false;
        }
    }
    return true;
}

static std::string srt_trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Parse the buffer into cues. Drives off the timing line ("-->"): the
// integer line just above it is the index when present, the lines below up
// to the next blank line are the text. Lines that are not part of a valid
// block are skipped. Returns true with out populated, false only on a NULL
// shaped buffer (empty out with a true return means no cue was found).
static bool srt_parse(const std::string & raw, std::vector<SrtCue> & out) {
    std::vector<std::string> lines = srt_split_lines(raw);

    size_t i = 0;
    while (i < lines.size()) {
        size_t arrow = lines[i].find("-->");
        if (arrow == std::string::npos) {
            i++;
            continue;
        }

        SrtCue cue;
        cue.index = 0;
        if (!srt_parse_time(lines[i].substr(0, arrow), cue.t0) || !srt_parse_time(lines[i].substr(arrow + 3), cue.t1)) {
            i++;
            continue;
        }

        // Recover the index from the previous line when it is a bare integer.
        if (i >= 1) {
            std::string prev = srt_trim(lines[i - 1]);
            size_t      k    = 0;
            int         idx  = 0;
            if (!prev.empty() && srt_read_uint(prev, k, idx) && k == prev.size()) {
                cue.index = idx;
            }
        }

        // Collect text lines until the next blank line or EOF.
        std::string text;
        size_t      j = i + 1;
        while (j < lines.size() && !srt_is_blank(lines[j])) {
            if (!text.empty()) {
                text.push_back(' ');
            }
            text += srt_trim(lines[j]);
            j++;
        }

        cue.text = srt_trim(text);
        out.push_back(cue);
        i = j;
    }
    return true;
}
