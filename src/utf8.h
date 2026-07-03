#pragma once
// utf8.h: portable UTF-8 boundary for Windows. Inside the project everything
// is UTF-8; this header bridges to the Windows-native UTF-16 APIs at the
// four places where the OS forces a recode: argv (CRT decodes from CP_ACP),
// fopen (CRT does too), any direct Win32 *A call (CreateFileA etc.), and
// text bytes arriving from stdin or files in a shell-chosen encoding.
// POSIX is UTF-8 by convention and every helper degrades to a passthrough.

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
// clang-format off
// windows.h must precede shellapi.h: shellapi.h pulls in types (HDROP,
// DECLSPEC_IMPORT, ...) declared by windows.h. SortIncludes would alphabetise
// the pair and break the build, so the off/on pragmas pin the order.
#    include <windows.h>
#    include <shellapi.h>
// clang-format on
#    include <string>
#endif

#if defined(_WIN32)
// Decode a UTF-8 C string to a wide string (UTF-16). Empty on null or empty
// input. Used by every Win32 path-taking call site (CreateFileW, MoveFileW
// and friends) wherever the project still needs the raw wide form.
static std::wstring utf8_to_wide(const char * s) {
    if (!s || !*s) {
        return std::wstring();
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    if (n <= 1) {
        return std::wstring();
    }
    std::wstring w((size_t) (n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    return w;
}
#endif

// First line of main(). Rebuilds argv as UTF-8 from the wide command line
// and switches console I/O to CP_UTF8 so fprintf with non-ASCII bytes
// renders correctly without an explicit chcp 65001. POSIX no-op.
//
// The replacement argv is allocated once and leaked for the process
// lifetime: the OS reclaims it on exit, and the top of main is the only
// site that would ever free it.
static void utf8_init(int * argc, char *** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    int      wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) {
        return;
    }

    char ** new_argv = (char **) malloc(sizeof(char *) * (size_t) (wargc + 1));
    for (int i = 0; i < wargc; i++) {
        int n       = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        new_argv[i] = (char *) malloc((size_t) n);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, new_argv[i], n, NULL, NULL);
    }
    new_argv[wargc] = NULL;
    LocalFree(wargv);

    *argc = wargc;
    *argv = new_argv;
#else
    (void) argc;
    (void) argv;
#endif
}

// UTF-8 aware fopen. POSIX passthrough. Windows decodes the UTF-8 path and
// mode to UTF-16 and calls _wfopen, the only fopen variant on MSVC that
// bypasses CP_ACP.
static FILE * utf8_fopen(const char * path, const char * mode) {
#if defined(_WIN32)
    std::wstring wpath = utf8_to_wide(path);
    std::wstring wmode = utf8_to_wide(mode);
    return _wfopen(wpath.c_str(), wmode.c_str());
#else
    return fopen(path, mode);
#endif
}

#if !defined(_WIN32)
#    include <string>
#endif

// Normalizes a text buffer read from stdin or a file to UTF-8 in place.
// Windows shells and editors hand bytes over in whatever encoding they
// default to: UTF-16 with BOM (PowerShell redirection, Notepad) or the
// ANSI codepage (cmd pipes, legacy editors). Both recode losslessly. A
// UTF-8 BOM is stripped on every platform, valid UTF-8 passes through
// untouched, and the ANSI decode only runs on bytes that fail UTF-8
// validation so it can never alter a well formed prompt.
static void utf8_normalize(std::string & s) {
    if (s.size() >= 3 && (unsigned char) s[0] == 0xEF && (unsigned char) s[1] == 0xBB && (unsigned char) s[2] == 0xBF) {
        s.erase(0, 3);
        return;
    }
#if defined(_WIN32)
    if (s.size() >= 2) {
        const unsigned char b0 = (unsigned char) s[0];
        const unsigned char b1 = (unsigned char) s[1];
        if ((b0 == 0xFF && b1 == 0xFE) || (b0 == 0xFE && b1 == 0xFF)) {
            if (b0 == 0xFE) {
                // UTF-16BE: swap every pair so the buffer reads as UTF-16LE.
                for (size_t i = 0; i + 1 < s.size(); i += 2) {
                    char t   = s[i];
                    s[i]     = s[i + 1];
                    s[i + 1] = t;
                }
            }
            const wchar_t * w = (const wchar_t *) (s.data() + 2);
            const int       n = (int) ((s.size() - 2) / sizeof(wchar_t));
            int             u = WideCharToMultiByte(CP_UTF8, 0, w, n, NULL, 0, NULL, NULL);
            std::string     out((size_t) u, '\0');
            WideCharToMultiByte(CP_UTF8, 0, w, n, &out[0], u, NULL, NULL);
            s.swap(out);
            return;
        }
    }
    if (s.empty() || MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), NULL, 0) > 0) {
        return;
    }
    int          n = MultiByteToWideChar(CP_ACP, 0, s.data(), (int) s.size(), NULL, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), (int) s.size(), &w[0], n);
    int         u = WideCharToMultiByte(CP_UTF8, 0, w.data(), n, NULL, 0, NULL, NULL);
    std::string out((size_t) u, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), n, &out[0], u, NULL, NULL);
    s.swap(out);
#endif
}
