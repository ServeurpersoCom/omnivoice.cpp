#pragma once
// duration-estimator.h: text duration estimation, byte-perfect mirror of
// omnivoice/utils/duration.py RuleDurationEstimator.
//
// Returns an estimated audio duration (in the same unit as ref_duration,
// typically audio tokens) for a target text given a reference text whose
// duration is known. Uses per-script phonetic weights and Unicode
// category fallbacks to handle 600+ scripts uniformly.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

// Top-level Unicode category classes used by the estimator.
// Generated from CPython 15.1.0 UCD over [0, 0x110000).
#define DUR_M 0  // Mark         (weight 0.0)
#define DUR_P 1  // Punctuation  (weight 0.5)
#define DUR_S 2  // Symbol       (weight 0.5)
#define DUR_Z 3  // Separator    (weight 0.2)
#define DUR_N 4  // Number       (weight 3.5)

struct DurationCatRange {
    uint32_t start;
    uint32_t end;
    uint8_t  cls;
};

static const DurationCatRange duration_cat_ranges[879] = {
    { 0x000020, 0x000020, DUR_Z },
    { 0x000021, 0x000023, DUR_P },
    { 0x000024, 0x000024, DUR_S },
    { 0x000025, 0x00002A, DUR_P },
    { 0x00002B, 0x00002B, DUR_S },
    { 0x00002C, 0x00002F, DUR_P },
    { 0x000030, 0x000039, DUR_N },
    { 0x00003A, 0x00003B, DUR_P },
    { 0x00003C, 0x00003E, DUR_S },
    { 0x00003F, 0x000040, DUR_P },
    { 0x00005B, 0x00005D, DUR_P },
    { 0x00005E, 0x00005E, DUR_S },
    { 0x00005F, 0x00005F, DUR_P },
    { 0x000060, 0x000060, DUR_S },
    { 0x00007B, 0x00007B, DUR_P },
    { 0x00007C, 0x00007C, DUR_S },
    { 0x00007D, 0x00007D, DUR_P },
    { 0x00007E, 0x00007E, DUR_S },
    { 0x0000A0, 0x0000A0, DUR_Z },
    { 0x0000A1, 0x0000A1, DUR_P },
    { 0x0000A2, 0x0000A6, DUR_S },
    { 0x0000A7, 0x0000A7, DUR_P },
    { 0x0000A8, 0x0000A9, DUR_S },
    { 0x0000AB, 0x0000AB, DUR_P },
    { 0x0000AC, 0x0000AC, DUR_S },
    { 0x0000AE, 0x0000B1, DUR_S },
    { 0x0000B2, 0x0000B3, DUR_N },
    { 0x0000B4, 0x0000B4, DUR_S },
    { 0x0000B6, 0x0000B7, DUR_P },
    { 0x0000B8, 0x0000B8, DUR_S },
    { 0x0000B9, 0x0000B9, DUR_N },
    { 0x0000BB, 0x0000BB, DUR_P },
    { 0x0000BC, 0x0000BE, DUR_N },
    { 0x0000BF, 0x0000BF, DUR_P },
    { 0x0000D7, 0x0000D7, DUR_S },
    { 0x0000F7, 0x0000F7, DUR_S },
    { 0x0002C2, 0x0002C5, DUR_S },
    { 0x0002D2, 0x0002DF, DUR_S },
    { 0x0002E5, 0x0002EB, DUR_S },
    { 0x0002ED, 0x0002ED, DUR_S },
    { 0x0002EF, 0x0002FF, DUR_S },
    { 0x000300, 0x00036F, DUR_M },
    { 0x000375, 0x000375, DUR_S },
    { 0x00037E, 0x00037E, DUR_P },
    { 0x000384, 0x000385, DUR_S },
    { 0x000387, 0x000387, DUR_P },
    { 0x0003F6, 0x0003F6, DUR_S },
    { 0x000482, 0x000482, DUR_S },
    { 0x000483, 0x000489, DUR_M },
    { 0x00055A, 0x00055F, DUR_P },
    { 0x000589, 0x00058A, DUR_P },
    { 0x00058D, 0x00058F, DUR_S },
    { 0x000591, 0x0005BD, DUR_M },
    { 0x0005BE, 0x0005BE, DUR_P },
    { 0x0005BF, 0x0005BF, DUR_M },
    { 0x0005C0, 0x0005C0, DUR_P },
    { 0x0005C1, 0x0005C2, DUR_M },
    { 0x0005C3, 0x0005C3, DUR_P },
    { 0x0005C4, 0x0005C5, DUR_M },
    { 0x0005C6, 0x0005C6, DUR_P },
    { 0x0005C7, 0x0005C7, DUR_M },
    { 0x0005F3, 0x0005F4, DUR_P },
    { 0x000606, 0x000608, DUR_S },
    { 0x000609, 0x00060A, DUR_P },
    { 0x00060B, 0x00060B, DUR_S },
    { 0x00060C, 0x00060D, DUR_P },
    { 0x00060E, 0x00060F, DUR_S },
    { 0x000610, 0x00061A, DUR_M },
    { 0x00061B, 0x00061B, DUR_P },
    { 0x00061D, 0x00061F, DUR_P },
    { 0x00064B, 0x00065F, DUR_M },
    { 0x000660, 0x000669, DUR_N },
    { 0x00066A, 0x00066D, DUR_P },
    { 0x000670, 0x000670, DUR_M },
    { 0x0006D4, 0x0006D4, DUR_P },
    { 0x0006D6, 0x0006DC, DUR_M },
    { 0x0006DE, 0x0006DE, DUR_S },
    { 0x0006DF, 0x0006E4, DUR_M },
    { 0x0006E7, 0x0006E8, DUR_M },
    { 0x0006E9, 0x0006E9, DUR_S },
    { 0x0006EA, 0x0006ED, DUR_M },
    { 0x0006F0, 0x0006F9, DUR_N },
    { 0x0006FD, 0x0006FE, DUR_S },
    { 0x000700, 0x00070D, DUR_P },
    { 0x000711, 0x000711, DUR_M },
    { 0x000730, 0x00074A, DUR_M },
    { 0x0007A6, 0x0007B0, DUR_M },
    { 0x0007C0, 0x0007C9, DUR_N },
    { 0x0007EB, 0x0007F3, DUR_M },
    { 0x0007F6, 0x0007F6, DUR_S },
    { 0x0007F7, 0x0007F9, DUR_P },
    { 0x0007FD, 0x0007FD, DUR_M },
    { 0x0007FE, 0x0007FF, DUR_S },
    { 0x000816, 0x000819, DUR_M },
    { 0x00081B, 0x000823, DUR_M },
    { 0x000825, 0x000827, DUR_M },
    { 0x000829, 0x00082D, DUR_M },
    { 0x000830, 0x00083E, DUR_P },
    { 0x000859, 0x00085B, DUR_M },
    { 0x00085E, 0x00085E, DUR_P },
    { 0x000888, 0x000888, DUR_S },
    { 0x000898, 0x00089F, DUR_M },
    { 0x0008CA, 0x0008E1, DUR_M },
    { 0x0008E3, 0x000903, DUR_M },
    { 0x00093A, 0x00093C, DUR_M },
    { 0x00093E, 0x00094F, DUR_M },
    { 0x000951, 0x000957, DUR_M },
    { 0x000962, 0x000963, DUR_M },
    { 0x000964, 0x000965, DUR_P },
    { 0x000966, 0x00096F, DUR_N },
    { 0x000970, 0x000970, DUR_P },
    { 0x000981, 0x000983, DUR_M },
    { 0x0009BC, 0x0009BC, DUR_M },
    { 0x0009BE, 0x0009C4, DUR_M },
    { 0x0009C7, 0x0009C8, DUR_M },
    { 0x0009CB, 0x0009CD, DUR_M },
    { 0x0009D7, 0x0009D7, DUR_M },
    { 0x0009E2, 0x0009E3, DUR_M },
    { 0x0009E6, 0x0009EF, DUR_N },
    { 0x0009F2, 0x0009F3, DUR_S },
    { 0x0009F4, 0x0009F9, DUR_N },
    { 0x0009FA, 0x0009FB, DUR_S },
    { 0x0009FD, 0x0009FD, DUR_P },
    { 0x0009FE, 0x0009FE, DUR_M },
    { 0x000A01, 0x000A03, DUR_M },
    { 0x000A3C, 0x000A3C, DUR_M },
    { 0x000A3E, 0x000A42, DUR_M },
    { 0x000A47, 0x000A48, DUR_M },
    { 0x000A4B, 0x000A4D, DUR_M },
    { 0x000A51, 0x000A51, DUR_M },
    { 0x000A66, 0x000A6F, DUR_N },
    { 0x000A70, 0x000A71, DUR_M },
    { 0x000A75, 0x000A75, DUR_M },
    { 0x000A76, 0x000A76, DUR_P },
    { 0x000A81, 0x000A83, DUR_M },
    { 0x000ABC, 0x000ABC, DUR_M },
    { 0x000ABE, 0x000AC5, DUR_M },
    { 0x000AC7, 0x000AC9, DUR_M },
    { 0x000ACB, 0x000ACD, DUR_M },
    { 0x000AE2, 0x000AE3, DUR_M },
    { 0x000AE6, 0x000AEF, DUR_N },
    { 0x000AF0, 0x000AF0, DUR_P },
    { 0x000AF1, 0x000AF1, DUR_S },
    { 0x000AFA, 0x000AFF, DUR_M },
    { 0x000B01, 0x000B03, DUR_M },
    { 0x000B3C, 0x000B3C, DUR_M },
    { 0x000B3E, 0x000B44, DUR_M },
    { 0x000B47, 0x000B48, DUR_M },
    { 0x000B4B, 0x000B4D, DUR_M },
    { 0x000B55, 0x000B57, DUR_M },
    { 0x000B62, 0x000B63, DUR_M },
    { 0x000B66, 0x000B6F, DUR_N },
    { 0x000B70, 0x000B70, DUR_S },
    { 0x000B72, 0x000B77, DUR_N },
    { 0x000B82, 0x000B82, DUR_M },
    { 0x000BBE, 0x000BC2, DUR_M },
    { 0x000BC6, 0x000BC8, DUR_M },
    { 0x000BCA, 0x000BCD, DUR_M },
    { 0x000BD7, 0x000BD7, DUR_M },
    { 0x000BE6, 0x000BF2, DUR_N },
    { 0x000BF3, 0x000BFA, DUR_S },
    { 0x000C00, 0x000C04, DUR_M },
    { 0x000C3C, 0x000C3C, DUR_M },
    { 0x000C3E, 0x000C44, DUR_M },
    { 0x000C46, 0x000C48, DUR_M },
    { 0x000C4A, 0x000C4D, DUR_M },
    { 0x000C55, 0x000C56, DUR_M },
    { 0x000C62, 0x000C63, DUR_M },
    { 0x000C66, 0x000C6F, DUR_N },
    { 0x000C77, 0x000C77, DUR_P },
    { 0x000C78, 0x000C7E, DUR_N },
    { 0x000C7F, 0x000C7F, DUR_S },
    { 0x000C81, 0x000C83, DUR_M },
    { 0x000C84, 0x000C84, DUR_P },
    { 0x000CBC, 0x000CBC, DUR_M },
    { 0x000CBE, 0x000CC4, DUR_M },
    { 0x000CC6, 0x000CC8, DUR_M },
    { 0x000CCA, 0x000CCD, DUR_M },
    { 0x000CD5, 0x000CD6, DUR_M },
    { 0x000CE2, 0x000CE3, DUR_M },
    { 0x000CE6, 0x000CEF, DUR_N },
    { 0x000CF3, 0x000CF3, DUR_M },
    { 0x000D00, 0x000D03, DUR_M },
    { 0x000D3B, 0x000D3C, DUR_M },
    { 0x000D3E, 0x000D44, DUR_M },
    { 0x000D46, 0x000D48, DUR_M },
    { 0x000D4A, 0x000D4D, DUR_M },
    { 0x000D4F, 0x000D4F, DUR_S },
    { 0x000D57, 0x000D57, DUR_M },
    { 0x000D58, 0x000D5E, DUR_N },
    { 0x000D62, 0x000D63, DUR_M },
    { 0x000D66, 0x000D78, DUR_N },
    { 0x000D79, 0x000D79, DUR_S },
    { 0x000D81, 0x000D83, DUR_M },
    { 0x000DCA, 0x000DCA, DUR_M },
    { 0x000DCF, 0x000DD4, DUR_M },
    { 0x000DD6, 0x000DD6, DUR_M },
    { 0x000DD8, 0x000DDF, DUR_M },
    { 0x000DE6, 0x000DEF, DUR_N },
    { 0x000DF2, 0x000DF3, DUR_M },
    { 0x000DF4, 0x000DF4, DUR_P },
    { 0x000E31, 0x000E31, DUR_M },
    { 0x000E34, 0x000E3A, DUR_M },
    { 0x000E3F, 0x000E3F, DUR_S },
    { 0x000E47, 0x000E4E, DUR_M },
    { 0x000E4F, 0x000E4F, DUR_P },
    { 0x000E50, 0x000E59, DUR_N },
    { 0x000E5A, 0x000E5B, DUR_P },
    { 0x000EB1, 0x000EB1, DUR_M },
    { 0x000EB4, 0x000EBC, DUR_M },
    { 0x000EC8, 0x000ECE, DUR_M },
    { 0x000ED0, 0x000ED9, DUR_N },
    { 0x000F01, 0x000F03, DUR_S },
    { 0x000F04, 0x000F12, DUR_P },
    { 0x000F13, 0x000F13, DUR_S },
    { 0x000F14, 0x000F14, DUR_P },
    { 0x000F15, 0x000F17, DUR_S },
    { 0x000F18, 0x000F19, DUR_M },
    { 0x000F1A, 0x000F1F, DUR_S },
    { 0x000F20, 0x000F33, DUR_N },
    { 0x000F34, 0x000F34, DUR_S },
    { 0x000F35, 0x000F35, DUR_M },
    { 0x000F36, 0x000F36, DUR_S },
    { 0x000F37, 0x000F37, DUR_M },
    { 0x000F38, 0x000F38, DUR_S },
    { 0x000F39, 0x000F39, DUR_M },
    { 0x000F3A, 0x000F3D, DUR_P },
    { 0x000F3E, 0x000F3F, DUR_M },
    { 0x000F71, 0x000F84, DUR_M },
    { 0x000F85, 0x000F85, DUR_P },
    { 0x000F86, 0x000F87, DUR_M },
    { 0x000F8D, 0x000F97, DUR_M },
    { 0x000F99, 0x000FBC, DUR_M },
    { 0x000FBE, 0x000FC5, DUR_S },
    { 0x000FC6, 0x000FC6, DUR_M },
    { 0x000FC7, 0x000FCC, DUR_S },
    { 0x000FCE, 0x000FCF, DUR_S },
    { 0x000FD0, 0x000FD4, DUR_P },
    { 0x000FD5, 0x000FD8, DUR_S },
    { 0x000FD9, 0x000FDA, DUR_P },
    { 0x00102B, 0x00103E, DUR_M },
    { 0x001040, 0x001049, DUR_N },
    { 0x00104A, 0x00104F, DUR_P },
    { 0x001056, 0x001059, DUR_M },
    { 0x00105E, 0x001060, DUR_M },
    { 0x001062, 0x001064, DUR_M },
    { 0x001067, 0x00106D, DUR_M },
    { 0x001071, 0x001074, DUR_M },
    { 0x001082, 0x00108D, DUR_M },
    { 0x00108F, 0x00108F, DUR_M },
    { 0x001090, 0x001099, DUR_N },
    { 0x00109A, 0x00109D, DUR_M },
    { 0x00109E, 0x00109F, DUR_S },
    { 0x0010FB, 0x0010FB, DUR_P },
    { 0x00135D, 0x00135F, DUR_M },
    { 0x001360, 0x001368, DUR_P },
    { 0x001369, 0x00137C, DUR_N },
    { 0x001390, 0x001399, DUR_S },
    { 0x001400, 0x001400, DUR_P },
    { 0x00166D, 0x00166D, DUR_S },
    { 0x00166E, 0x00166E, DUR_P },
    { 0x001680, 0x001680, DUR_Z },
    { 0x00169B, 0x00169C, DUR_P },
    { 0x0016EB, 0x0016ED, DUR_P },
    { 0x0016EE, 0x0016F0, DUR_N },
    { 0x001712, 0x001715, DUR_M },
    { 0x001732, 0x001734, DUR_M },
    { 0x001735, 0x001736, DUR_P },
    { 0x001752, 0x001753, DUR_M },
    { 0x001772, 0x001773, DUR_M },
    { 0x0017B4, 0x0017D3, DUR_M },
    { 0x0017D4, 0x0017D6, DUR_P },
    { 0x0017D8, 0x0017DA, DUR_P },
    { 0x0017DB, 0x0017DB, DUR_S },
    { 0x0017DD, 0x0017DD, DUR_M },
    { 0x0017E0, 0x0017E9, DUR_N },
    { 0x0017F0, 0x0017F9, DUR_N },
    { 0x001800, 0x00180A, DUR_P },
    { 0x00180B, 0x00180D, DUR_M },
    { 0x00180F, 0x00180F, DUR_M },
    { 0x001810, 0x001819, DUR_N },
    { 0x001885, 0x001886, DUR_M },
    { 0x0018A9, 0x0018A9, DUR_M },
    { 0x001920, 0x00192B, DUR_M },
    { 0x001930, 0x00193B, DUR_M },
    { 0x001940, 0x001940, DUR_S },
    { 0x001944, 0x001945, DUR_P },
    { 0x001946, 0x00194F, DUR_N },
    { 0x0019D0, 0x0019DA, DUR_N },
    { 0x0019DE, 0x0019FF, DUR_S },
    { 0x001A17, 0x001A1B, DUR_M },
    { 0x001A1E, 0x001A1F, DUR_P },
    { 0x001A55, 0x001A5E, DUR_M },
    { 0x001A60, 0x001A7C, DUR_M },
    { 0x001A7F, 0x001A7F, DUR_M },
    { 0x001A80, 0x001A89, DUR_N },
    { 0x001A90, 0x001A99, DUR_N },
    { 0x001AA0, 0x001AA6, DUR_P },
    { 0x001AA8, 0x001AAD, DUR_P },
    { 0x001AB0, 0x001ACE, DUR_M },
    { 0x001B00, 0x001B04, DUR_M },
    { 0x001B34, 0x001B44, DUR_M },
    { 0x001B50, 0x001B59, DUR_N },
    { 0x001B5A, 0x001B60, DUR_P },
    { 0x001B61, 0x001B6A, DUR_S },
    { 0x001B6B, 0x001B73, DUR_M },
    { 0x001B74, 0x001B7C, DUR_S },
    { 0x001B7D, 0x001B7E, DUR_P },
    { 0x001B80, 0x001B82, DUR_M },
    { 0x001BA1, 0x001BAD, DUR_M },
    { 0x001BB0, 0x001BB9, DUR_N },
    { 0x001BE6, 0x001BF3, DUR_M },
    { 0x001BFC, 0x001BFF, DUR_P },
    { 0x001C24, 0x001C37, DUR_M },
    { 0x001C3B, 0x001C3F, DUR_P },
    { 0x001C40, 0x001C49, DUR_N },
    { 0x001C50, 0x001C59, DUR_N },
    { 0x001C7E, 0x001C7F, DUR_P },
    { 0x001CC0, 0x001CC7, DUR_P },
    { 0x001CD0, 0x001CD2, DUR_M },
    { 0x001CD3, 0x001CD3, DUR_P },
    { 0x001CD4, 0x001CE8, DUR_M },
    { 0x001CED, 0x001CED, DUR_M },
    { 0x001CF4, 0x001CF4, DUR_M },
    { 0x001CF7, 0x001CF9, DUR_M },
    { 0x001DC0, 0x001DFF, DUR_M },
    { 0x001FBD, 0x001FBD, DUR_S },
    { 0x001FBF, 0x001FC1, DUR_S },
    { 0x001FCD, 0x001FCF, DUR_S },
    { 0x001FDD, 0x001FDF, DUR_S },
    { 0x001FED, 0x001FEF, DUR_S },
    { 0x001FFD, 0x001FFE, DUR_S },
    { 0x002000, 0x00200A, DUR_Z },
    { 0x002010, 0x002027, DUR_P },
    { 0x002028, 0x002029, DUR_Z },
    { 0x00202F, 0x00202F, DUR_Z },
    { 0x002030, 0x002043, DUR_P },
    { 0x002044, 0x002044, DUR_S },
    { 0x002045, 0x002051, DUR_P },
    { 0x002052, 0x002052, DUR_S },
    { 0x002053, 0x00205E, DUR_P },
    { 0x00205F, 0x00205F, DUR_Z },
    { 0x002070, 0x002070, DUR_N },
    { 0x002074, 0x002079, DUR_N },
    { 0x00207A, 0x00207C, DUR_S },
    { 0x00207D, 0x00207E, DUR_P },
    { 0x002080, 0x002089, DUR_N },
    { 0x00208A, 0x00208C, DUR_S },
    { 0x00208D, 0x00208E, DUR_P },
    { 0x0020A0, 0x0020C0, DUR_S },
    { 0x0020D0, 0x0020F0, DUR_M },
    { 0x002100, 0x002101, DUR_S },
    { 0x002103, 0x002106, DUR_S },
    { 0x002108, 0x002109, DUR_S },
    { 0x002114, 0x002114, DUR_S },
    { 0x002116, 0x002118, DUR_S },
    { 0x00211E, 0x002123, DUR_S },
    { 0x002125, 0x002125, DUR_S },
    { 0x002127, 0x002127, DUR_S },
    { 0x002129, 0x002129, DUR_S },
    { 0x00212E, 0x00212E, DUR_S },
    { 0x00213A, 0x00213B, DUR_S },
    { 0x002140, 0x002144, DUR_S },
    { 0x00214A, 0x00214D, DUR_S },
    { 0x00214F, 0x00214F, DUR_S },
    { 0x002150, 0x002182, DUR_N },
    { 0x002185, 0x002189, DUR_N },
    { 0x00218A, 0x00218B, DUR_S },
    { 0x002190, 0x002307, DUR_S },
    { 0x002308, 0x00230B, DUR_P },
    { 0x00230C, 0x002328, DUR_S },
    { 0x002329, 0x00232A, DUR_P },
    { 0x00232B, 0x002426, DUR_S },
    { 0x002440, 0x00244A, DUR_S },
    { 0x002460, 0x00249B, DUR_N },
    { 0x00249C, 0x0024E9, DUR_S },
    { 0x0024EA, 0x0024FF, DUR_N },
    { 0x002500, 0x002767, DUR_S },
    { 0x002768, 0x002775, DUR_P },
    { 0x002776, 0x002793, DUR_N },
    { 0x002794, 0x0027C4, DUR_S },
    { 0x0027C5, 0x0027C6, DUR_P },
    { 0x0027C7, 0x0027E5, DUR_S },
    { 0x0027E6, 0x0027EF, DUR_P },
    { 0x0027F0, 0x002982, DUR_S },
    { 0x002983, 0x002998, DUR_P },
    { 0x002999, 0x0029D7, DUR_S },
    { 0x0029D8, 0x0029DB, DUR_P },
    { 0x0029DC, 0x0029FB, DUR_S },
    { 0x0029FC, 0x0029FD, DUR_P },
    { 0x0029FE, 0x002B73, DUR_S },
    { 0x002B76, 0x002B95, DUR_S },
    { 0x002B97, 0x002BFF, DUR_S },
    { 0x002CE5, 0x002CEA, DUR_S },
    { 0x002CEF, 0x002CF1, DUR_M },
    { 0x002CF9, 0x002CFC, DUR_P },
    { 0x002CFD, 0x002CFD, DUR_N },
    { 0x002CFE, 0x002CFF, DUR_P },
    { 0x002D70, 0x002D70, DUR_P },
    { 0x002D7F, 0x002D7F, DUR_M },
    { 0x002DE0, 0x002DFF, DUR_M },
    { 0x002E00, 0x002E2E, DUR_P },
    { 0x002E30, 0x002E4F, DUR_P },
    { 0x002E50, 0x002E51, DUR_S },
    { 0x002E52, 0x002E5D, DUR_P },
    { 0x002E80, 0x002E99, DUR_S },
    { 0x002E9B, 0x002EF3, DUR_S },
    { 0x002F00, 0x002FD5, DUR_S },
    { 0x002FF0, 0x002FFF, DUR_S },
    { 0x003000, 0x003000, DUR_Z },
    { 0x003001, 0x003003, DUR_P },
    { 0x003004, 0x003004, DUR_S },
    { 0x003007, 0x003007, DUR_N },
    { 0x003008, 0x003011, DUR_P },
    { 0x003012, 0x003013, DUR_S },
    { 0x003014, 0x00301F, DUR_P },
    { 0x003020, 0x003020, DUR_S },
    { 0x003021, 0x003029, DUR_N },
    { 0x00302A, 0x00302F, DUR_M },
    { 0x003030, 0x003030, DUR_P },
    { 0x003036, 0x003037, DUR_S },
    { 0x003038, 0x00303A, DUR_N },
    { 0x00303D, 0x00303D, DUR_P },
    { 0x00303E, 0x00303F, DUR_S },
    { 0x003099, 0x00309A, DUR_M },
    { 0x00309B, 0x00309C, DUR_S },
    { 0x0030A0, 0x0030A0, DUR_P },
    { 0x0030FB, 0x0030FB, DUR_P },
    { 0x003190, 0x003191, DUR_S },
    { 0x003192, 0x003195, DUR_N },
    { 0x003196, 0x00319F, DUR_S },
    { 0x0031C0, 0x0031E3, DUR_S },
    { 0x0031EF, 0x0031EF, DUR_S },
    { 0x003200, 0x00321E, DUR_S },
    { 0x003220, 0x003229, DUR_N },
    { 0x00322A, 0x003247, DUR_S },
    { 0x003248, 0x00324F, DUR_N },
    { 0x003250, 0x003250, DUR_S },
    { 0x003251, 0x00325F, DUR_N },
    { 0x003260, 0x00327F, DUR_S },
    { 0x003280, 0x003289, DUR_N },
    { 0x00328A, 0x0032B0, DUR_S },
    { 0x0032B1, 0x0032BF, DUR_N },
    { 0x0032C0, 0x0033FF, DUR_S },
    { 0x004DC0, 0x004DFF, DUR_S },
    { 0x00A490, 0x00A4C6, DUR_S },
    { 0x00A4FE, 0x00A4FF, DUR_P },
    { 0x00A60D, 0x00A60F, DUR_P },
    { 0x00A620, 0x00A629, DUR_N },
    { 0x00A66F, 0x00A672, DUR_M },
    { 0x00A673, 0x00A673, DUR_P },
    { 0x00A674, 0x00A67D, DUR_M },
    { 0x00A67E, 0x00A67E, DUR_P },
    { 0x00A69E, 0x00A69F, DUR_M },
    { 0x00A6E6, 0x00A6EF, DUR_N },
    { 0x00A6F0, 0x00A6F1, DUR_M },
    { 0x00A6F2, 0x00A6F7, DUR_P },
    { 0x00A700, 0x00A716, DUR_S },
    { 0x00A720, 0x00A721, DUR_S },
    { 0x00A789, 0x00A78A, DUR_S },
    { 0x00A802, 0x00A802, DUR_M },
    { 0x00A806, 0x00A806, DUR_M },
    { 0x00A80B, 0x00A80B, DUR_M },
    { 0x00A823, 0x00A827, DUR_M },
    { 0x00A828, 0x00A82B, DUR_S },
    { 0x00A82C, 0x00A82C, DUR_M },
    { 0x00A830, 0x00A835, DUR_N },
    { 0x00A836, 0x00A839, DUR_S },
    { 0x00A874, 0x00A877, DUR_P },
    { 0x00A880, 0x00A881, DUR_M },
    { 0x00A8B4, 0x00A8C5, DUR_M },
    { 0x00A8CE, 0x00A8CF, DUR_P },
    { 0x00A8D0, 0x00A8D9, DUR_N },
    { 0x00A8E0, 0x00A8F1, DUR_M },
    { 0x00A8F8, 0x00A8FA, DUR_P },
    { 0x00A8FC, 0x00A8FC, DUR_P },
    { 0x00A8FF, 0x00A8FF, DUR_M },
    { 0x00A900, 0x00A909, DUR_N },
    { 0x00A926, 0x00A92D, DUR_M },
    { 0x00A92E, 0x00A92F, DUR_P },
    { 0x00A947, 0x00A953, DUR_M },
    { 0x00A95F, 0x00A95F, DUR_P },
    { 0x00A980, 0x00A983, DUR_M },
    { 0x00A9B3, 0x00A9C0, DUR_M },
    { 0x00A9C1, 0x00A9CD, DUR_P },
    { 0x00A9D0, 0x00A9D9, DUR_N },
    { 0x00A9DE, 0x00A9DF, DUR_P },
    { 0x00A9E5, 0x00A9E5, DUR_M },
    { 0x00A9F0, 0x00A9F9, DUR_N },
    { 0x00AA29, 0x00AA36, DUR_M },
    { 0x00AA43, 0x00AA43, DUR_M },
    { 0x00AA4C, 0x00AA4D, DUR_M },
    { 0x00AA50, 0x00AA59, DUR_N },
    { 0x00AA5C, 0x00AA5F, DUR_P },
    { 0x00AA77, 0x00AA79, DUR_S },
    { 0x00AA7B, 0x00AA7D, DUR_M },
    { 0x00AAB0, 0x00AAB0, DUR_M },
    { 0x00AAB2, 0x00AAB4, DUR_M },
    { 0x00AAB7, 0x00AAB8, DUR_M },
    { 0x00AABE, 0x00AABF, DUR_M },
    { 0x00AAC1, 0x00AAC1, DUR_M },
    { 0x00AADE, 0x00AADF, DUR_P },
    { 0x00AAEB, 0x00AAEF, DUR_M },
    { 0x00AAF0, 0x00AAF1, DUR_P },
    { 0x00AAF5, 0x00AAF6, DUR_M },
    { 0x00AB5B, 0x00AB5B, DUR_S },
    { 0x00AB6A, 0x00AB6B, DUR_S },
    { 0x00ABE3, 0x00ABEA, DUR_M },
    { 0x00ABEB, 0x00ABEB, DUR_P },
    { 0x00ABEC, 0x00ABED, DUR_M },
    { 0x00ABF0, 0x00ABF9, DUR_N },
    { 0x00FB1E, 0x00FB1E, DUR_M },
    { 0x00FB29, 0x00FB29, DUR_S },
    { 0x00FBB2, 0x00FBC2, DUR_S },
    { 0x00FD3E, 0x00FD3F, DUR_P },
    { 0x00FD40, 0x00FD4F, DUR_S },
    { 0x00FDCF, 0x00FDCF, DUR_S },
    { 0x00FDFC, 0x00FDFF, DUR_S },
    { 0x00FE00, 0x00FE0F, DUR_M },
    { 0x00FE10, 0x00FE19, DUR_P },
    { 0x00FE20, 0x00FE2F, DUR_M },
    { 0x00FE30, 0x00FE52, DUR_P },
    { 0x00FE54, 0x00FE61, DUR_P },
    { 0x00FE62, 0x00FE62, DUR_S },
    { 0x00FE63, 0x00FE63, DUR_P },
    { 0x00FE64, 0x00FE66, DUR_S },
    { 0x00FE68, 0x00FE68, DUR_P },
    { 0x00FE69, 0x00FE69, DUR_S },
    { 0x00FE6A, 0x00FE6B, DUR_P },
    { 0x00FF01, 0x00FF03, DUR_P },
    { 0x00FF04, 0x00FF04, DUR_S },
    { 0x00FF05, 0x00FF0A, DUR_P },
    { 0x00FF0B, 0x00FF0B, DUR_S },
    { 0x00FF0C, 0x00FF0F, DUR_P },
    { 0x00FF10, 0x00FF19, DUR_N },
    { 0x00FF1A, 0x00FF1B, DUR_P },
    { 0x00FF1C, 0x00FF1E, DUR_S },
    { 0x00FF1F, 0x00FF20, DUR_P },
    { 0x00FF3B, 0x00FF3D, DUR_P },
    { 0x00FF3E, 0x00FF3E, DUR_S },
    { 0x00FF3F, 0x00FF3F, DUR_P },
    { 0x00FF40, 0x00FF40, DUR_S },
    { 0x00FF5B, 0x00FF5B, DUR_P },
    { 0x00FF5C, 0x00FF5C, DUR_S },
    { 0x00FF5D, 0x00FF5D, DUR_P },
    { 0x00FF5E, 0x00FF5E, DUR_S },
    { 0x00FF5F, 0x00FF65, DUR_P },
    { 0x00FFE0, 0x00FFE6, DUR_S },
    { 0x00FFE8, 0x00FFEE, DUR_S },
    { 0x00FFFC, 0x00FFFD, DUR_S },
    { 0x010100, 0x010102, DUR_P },
    { 0x010107, 0x010133, DUR_N },
    { 0x010137, 0x01013F, DUR_S },
    { 0x010140, 0x010178, DUR_N },
    { 0x010179, 0x010189, DUR_S },
    { 0x01018A, 0x01018B, DUR_N },
    { 0x01018C, 0x01018E, DUR_S },
    { 0x010190, 0x01019C, DUR_S },
    { 0x0101A0, 0x0101A0, DUR_S },
    { 0x0101D0, 0x0101FC, DUR_S },
    { 0x0101FD, 0x0101FD, DUR_M },
    { 0x0102E0, 0x0102E0, DUR_M },
    { 0x0102E1, 0x0102FB, DUR_N },
    { 0x010320, 0x010323, DUR_N },
    { 0x010341, 0x010341, DUR_N },
    { 0x01034A, 0x01034A, DUR_N },
    { 0x010376, 0x01037A, DUR_M },
    { 0x01039F, 0x01039F, DUR_P },
    { 0x0103D0, 0x0103D0, DUR_P },
    { 0x0103D1, 0x0103D5, DUR_N },
    { 0x0104A0, 0x0104A9, DUR_N },
    { 0x01056F, 0x01056F, DUR_P },
    { 0x010857, 0x010857, DUR_P },
    { 0x010858, 0x01085F, DUR_N },
    { 0x010877, 0x010878, DUR_S },
    { 0x010879, 0x01087F, DUR_N },
    { 0x0108A7, 0x0108AF, DUR_N },
    { 0x0108FB, 0x0108FF, DUR_N },
    { 0x010916, 0x01091B, DUR_N },
    { 0x01091F, 0x01091F, DUR_P },
    { 0x01093F, 0x01093F, DUR_P },
    { 0x0109BC, 0x0109BD, DUR_N },
    { 0x0109C0, 0x0109CF, DUR_N },
    { 0x0109D2, 0x0109FF, DUR_N },
    { 0x010A01, 0x010A03, DUR_M },
    { 0x010A05, 0x010A06, DUR_M },
    { 0x010A0C, 0x010A0F, DUR_M },
    { 0x010A38, 0x010A3A, DUR_M },
    { 0x010A3F, 0x010A3F, DUR_M },
    { 0x010A40, 0x010A48, DUR_N },
    { 0x010A50, 0x010A58, DUR_P },
    { 0x010A7D, 0x010A7E, DUR_N },
    { 0x010A7F, 0x010A7F, DUR_P },
    { 0x010A9D, 0x010A9F, DUR_N },
    { 0x010AC8, 0x010AC8, DUR_S },
    { 0x010AE5, 0x010AE6, DUR_M },
    { 0x010AEB, 0x010AEF, DUR_N },
    { 0x010AF0, 0x010AF6, DUR_P },
    { 0x010B39, 0x010B3F, DUR_P },
    { 0x010B58, 0x010B5F, DUR_N },
    { 0x010B78, 0x010B7F, DUR_N },
    { 0x010B99, 0x010B9C, DUR_P },
    { 0x010BA9, 0x010BAF, DUR_N },
    { 0x010CFA, 0x010CFF, DUR_N },
    { 0x010D24, 0x010D27, DUR_M },
    { 0x010D30, 0x010D39, DUR_N },
    { 0x010E60, 0x010E7E, DUR_N },
    { 0x010EAB, 0x010EAC, DUR_M },
    { 0x010EAD, 0x010EAD, DUR_P },
    { 0x010EFD, 0x010EFF, DUR_M },
    { 0x010F1D, 0x010F26, DUR_N },
    { 0x010F46, 0x010F50, DUR_M },
    { 0x010F51, 0x010F54, DUR_N },
    { 0x010F55, 0x010F59, DUR_P },
    { 0x010F82, 0x010F85, DUR_M },
    { 0x010F86, 0x010F89, DUR_P },
    { 0x010FC5, 0x010FCB, DUR_N },
    { 0x011000, 0x011002, DUR_M },
    { 0x011038, 0x011046, DUR_M },
    { 0x011047, 0x01104D, DUR_P },
    { 0x011052, 0x01106F, DUR_N },
    { 0x011070, 0x011070, DUR_M },
    { 0x011073, 0x011074, DUR_M },
    { 0x01107F, 0x011082, DUR_M },
    { 0x0110B0, 0x0110BA, DUR_M },
    { 0x0110BB, 0x0110BC, DUR_P },
    { 0x0110BE, 0x0110C1, DUR_P },
    { 0x0110C2, 0x0110C2, DUR_M },
    { 0x0110F0, 0x0110F9, DUR_N },
    { 0x011100, 0x011102, DUR_M },
    { 0x011127, 0x011134, DUR_M },
    { 0x011136, 0x01113F, DUR_N },
    { 0x011140, 0x011143, DUR_P },
    { 0x011145, 0x011146, DUR_M },
    { 0x011173, 0x011173, DUR_M },
    { 0x011174, 0x011175, DUR_P },
    { 0x011180, 0x011182, DUR_M },
    { 0x0111B3, 0x0111C0, DUR_M },
    { 0x0111C5, 0x0111C8, DUR_P },
    { 0x0111C9, 0x0111CC, DUR_M },
    { 0x0111CD, 0x0111CD, DUR_P },
    { 0x0111CE, 0x0111CF, DUR_M },
    { 0x0111D0, 0x0111D9, DUR_N },
    { 0x0111DB, 0x0111DB, DUR_P },
    { 0x0111DD, 0x0111DF, DUR_P },
    { 0x0111E1, 0x0111F4, DUR_N },
    { 0x01122C, 0x011237, DUR_M },
    { 0x011238, 0x01123D, DUR_P },
    { 0x01123E, 0x01123E, DUR_M },
    { 0x011241, 0x011241, DUR_M },
    { 0x0112A9, 0x0112A9, DUR_P },
    { 0x0112DF, 0x0112EA, DUR_M },
    { 0x0112F0, 0x0112F9, DUR_N },
    { 0x011300, 0x011303, DUR_M },
    { 0x01133B, 0x01133C, DUR_M },
    { 0x01133E, 0x011344, DUR_M },
    { 0x011347, 0x011348, DUR_M },
    { 0x01134B, 0x01134D, DUR_M },
    { 0x011357, 0x011357, DUR_M },
    { 0x011362, 0x011363, DUR_M },
    { 0x011366, 0x01136C, DUR_M },
    { 0x011370, 0x011374, DUR_M },
    { 0x011435, 0x011446, DUR_M },
    { 0x01144B, 0x01144F, DUR_P },
    { 0x011450, 0x011459, DUR_N },
    { 0x01145A, 0x01145B, DUR_P },
    { 0x01145D, 0x01145D, DUR_P },
    { 0x01145E, 0x01145E, DUR_M },
    { 0x0114B0, 0x0114C3, DUR_M },
    { 0x0114C6, 0x0114C6, DUR_P },
    { 0x0114D0, 0x0114D9, DUR_N },
    { 0x0115AF, 0x0115B5, DUR_M },
    { 0x0115B8, 0x0115C0, DUR_M },
    { 0x0115C1, 0x0115D7, DUR_P },
    { 0x0115DC, 0x0115DD, DUR_M },
    { 0x011630, 0x011640, DUR_M },
    { 0x011641, 0x011643, DUR_P },
    { 0x011650, 0x011659, DUR_N },
    { 0x011660, 0x01166C, DUR_P },
    { 0x0116AB, 0x0116B7, DUR_M },
    { 0x0116B9, 0x0116B9, DUR_P },
    { 0x0116C0, 0x0116C9, DUR_N },
    { 0x01171D, 0x01172B, DUR_M },
    { 0x011730, 0x01173B, DUR_N },
    { 0x01173C, 0x01173E, DUR_P },
    { 0x01173F, 0x01173F, DUR_S },
    { 0x01182C, 0x01183A, DUR_M },
    { 0x01183B, 0x01183B, DUR_P },
    { 0x0118E0, 0x0118F2, DUR_N },
    { 0x011930, 0x011935, DUR_M },
    { 0x011937, 0x011938, DUR_M },
    { 0x01193B, 0x01193E, DUR_M },
    { 0x011940, 0x011940, DUR_M },
    { 0x011942, 0x011943, DUR_M },
    { 0x011944, 0x011946, DUR_P },
    { 0x011950, 0x011959, DUR_N },
    { 0x0119D1, 0x0119D7, DUR_M },
    { 0x0119DA, 0x0119E0, DUR_M },
    { 0x0119E2, 0x0119E2, DUR_P },
    { 0x0119E4, 0x0119E4, DUR_M },
    { 0x011A01, 0x011A0A, DUR_M },
    { 0x011A33, 0x011A39, DUR_M },
    { 0x011A3B, 0x011A3E, DUR_M },
    { 0x011A3F, 0x011A46, DUR_P },
    { 0x011A47, 0x011A47, DUR_M },
    { 0x011A51, 0x011A5B, DUR_M },
    { 0x011A8A, 0x011A99, DUR_M },
    { 0x011A9A, 0x011A9C, DUR_P },
    { 0x011A9E, 0x011AA2, DUR_P },
    { 0x011B00, 0x011B09, DUR_P },
    { 0x011C2F, 0x011C36, DUR_M },
    { 0x011C38, 0x011C3F, DUR_M },
    { 0x011C41, 0x011C45, DUR_P },
    { 0x011C50, 0x011C6C, DUR_N },
    { 0x011C70, 0x011C71, DUR_P },
    { 0x011C92, 0x011CA7, DUR_M },
    { 0x011CA9, 0x011CB6, DUR_M },
    { 0x011D31, 0x011D36, DUR_M },
    { 0x011D3A, 0x011D3A, DUR_M },
    { 0x011D3C, 0x011D3D, DUR_M },
    { 0x011D3F, 0x011D45, DUR_M },
    { 0x011D47, 0x011D47, DUR_M },
    { 0x011D50, 0x011D59, DUR_N },
    { 0x011D8A, 0x011D8E, DUR_M },
    { 0x011D90, 0x011D91, DUR_M },
    { 0x011D93, 0x011D97, DUR_M },
    { 0x011DA0, 0x011DA9, DUR_N },
    { 0x011EF3, 0x011EF6, DUR_M },
    { 0x011EF7, 0x011EF8, DUR_P },
    { 0x011F00, 0x011F01, DUR_M },
    { 0x011F03, 0x011F03, DUR_M },
    { 0x011F34, 0x011F3A, DUR_M },
    { 0x011F3E, 0x011F42, DUR_M },
    { 0x011F43, 0x011F4F, DUR_P },
    { 0x011F50, 0x011F59, DUR_N },
    { 0x011FC0, 0x011FD4, DUR_N },
    { 0x011FD5, 0x011FF1, DUR_S },
    { 0x011FFF, 0x011FFF, DUR_P },
    { 0x012400, 0x01246E, DUR_N },
    { 0x012470, 0x012474, DUR_P },
    { 0x012FF1, 0x012FF2, DUR_P },
    { 0x013440, 0x013440, DUR_M },
    { 0x013447, 0x013455, DUR_M },
    { 0x016A60, 0x016A69, DUR_N },
    { 0x016A6E, 0x016A6F, DUR_P },
    { 0x016AC0, 0x016AC9, DUR_N },
    { 0x016AF0, 0x016AF4, DUR_M },
    { 0x016AF5, 0x016AF5, DUR_P },
    { 0x016B30, 0x016B36, DUR_M },
    { 0x016B37, 0x016B3B, DUR_P },
    { 0x016B3C, 0x016B3F, DUR_S },
    { 0x016B44, 0x016B44, DUR_P },
    { 0x016B45, 0x016B45, DUR_S },
    { 0x016B50, 0x016B59, DUR_N },
    { 0x016B5B, 0x016B61, DUR_N },
    { 0x016E80, 0x016E96, DUR_N },
    { 0x016E97, 0x016E9A, DUR_P },
    { 0x016F4F, 0x016F4F, DUR_M },
    { 0x016F51, 0x016F87, DUR_M },
    { 0x016F8F, 0x016F92, DUR_M },
    { 0x016FE2, 0x016FE2, DUR_P },
    { 0x016FE4, 0x016FE4, DUR_M },
    { 0x016FF0, 0x016FF1, DUR_M },
    { 0x01BC9C, 0x01BC9C, DUR_S },
    { 0x01BC9D, 0x01BC9E, DUR_M },
    { 0x01BC9F, 0x01BC9F, DUR_P },
    { 0x01CF00, 0x01CF2D, DUR_M },
    { 0x01CF30, 0x01CF46, DUR_M },
    { 0x01CF50, 0x01CFC3, DUR_S },
    { 0x01D000, 0x01D0F5, DUR_S },
    { 0x01D100, 0x01D126, DUR_S },
    { 0x01D129, 0x01D164, DUR_S },
    { 0x01D165, 0x01D169, DUR_M },
    { 0x01D16A, 0x01D16C, DUR_S },
    { 0x01D16D, 0x01D172, DUR_M },
    { 0x01D17B, 0x01D182, DUR_M },
    { 0x01D183, 0x01D184, DUR_S },
    { 0x01D185, 0x01D18B, DUR_M },
    { 0x01D18C, 0x01D1A9, DUR_S },
    { 0x01D1AA, 0x01D1AD, DUR_M },
    { 0x01D1AE, 0x01D1EA, DUR_S },
    { 0x01D200, 0x01D241, DUR_S },
    { 0x01D242, 0x01D244, DUR_M },
    { 0x01D245, 0x01D245, DUR_S },
    { 0x01D2C0, 0x01D2D3, DUR_N },
    { 0x01D2E0, 0x01D2F3, DUR_N },
    { 0x01D300, 0x01D356, DUR_S },
    { 0x01D360, 0x01D378, DUR_N },
    { 0x01D6C1, 0x01D6C1, DUR_S },
    { 0x01D6DB, 0x01D6DB, DUR_S },
    { 0x01D6FB, 0x01D6FB, DUR_S },
    { 0x01D715, 0x01D715, DUR_S },
    { 0x01D735, 0x01D735, DUR_S },
    { 0x01D74F, 0x01D74F, DUR_S },
    { 0x01D76F, 0x01D76F, DUR_S },
    { 0x01D789, 0x01D789, DUR_S },
    { 0x01D7A9, 0x01D7A9, DUR_S },
    { 0x01D7C3, 0x01D7C3, DUR_S },
    { 0x01D7CE, 0x01D7FF, DUR_N },
    { 0x01D800, 0x01D9FF, DUR_S },
    { 0x01DA00, 0x01DA36, DUR_M },
    { 0x01DA37, 0x01DA3A, DUR_S },
    { 0x01DA3B, 0x01DA6C, DUR_M },
    { 0x01DA6D, 0x01DA74, DUR_S },
    { 0x01DA75, 0x01DA75, DUR_M },
    { 0x01DA76, 0x01DA83, DUR_S },
    { 0x01DA84, 0x01DA84, DUR_M },
    { 0x01DA85, 0x01DA86, DUR_S },
    { 0x01DA87, 0x01DA8B, DUR_P },
    { 0x01DA9B, 0x01DA9F, DUR_M },
    { 0x01DAA1, 0x01DAAF, DUR_M },
    { 0x01E000, 0x01E006, DUR_M },
    { 0x01E008, 0x01E018, DUR_M },
    { 0x01E01B, 0x01E021, DUR_M },
    { 0x01E023, 0x01E024, DUR_M },
    { 0x01E026, 0x01E02A, DUR_M },
    { 0x01E08F, 0x01E08F, DUR_M },
    { 0x01E130, 0x01E136, DUR_M },
    { 0x01E140, 0x01E149, DUR_N },
    { 0x01E14F, 0x01E14F, DUR_S },
    { 0x01E2AE, 0x01E2AE, DUR_M },
    { 0x01E2EC, 0x01E2EF, DUR_M },
    { 0x01E2F0, 0x01E2F9, DUR_N },
    { 0x01E2FF, 0x01E2FF, DUR_S },
    { 0x01E4EC, 0x01E4EF, DUR_M },
    { 0x01E4F0, 0x01E4F9, DUR_N },
    { 0x01E8C7, 0x01E8CF, DUR_N },
    { 0x01E8D0, 0x01E8D6, DUR_M },
    { 0x01E944, 0x01E94A, DUR_M },
    { 0x01E950, 0x01E959, DUR_N },
    { 0x01E95E, 0x01E95F, DUR_P },
    { 0x01EC71, 0x01ECAB, DUR_N },
    { 0x01ECAC, 0x01ECAC, DUR_S },
    { 0x01ECAD, 0x01ECAF, DUR_N },
    { 0x01ECB0, 0x01ECB0, DUR_S },
    { 0x01ECB1, 0x01ECB4, DUR_N },
    { 0x01ED01, 0x01ED2D, DUR_N },
    { 0x01ED2E, 0x01ED2E, DUR_S },
    { 0x01ED2F, 0x01ED3D, DUR_N },
    { 0x01EEF0, 0x01EEF1, DUR_S },
    { 0x01F000, 0x01F02B, DUR_S },
    { 0x01F030, 0x01F093, DUR_S },
    { 0x01F0A0, 0x01F0AE, DUR_S },
    { 0x01F0B1, 0x01F0BF, DUR_S },
    { 0x01F0C1, 0x01F0CF, DUR_S },
    { 0x01F0D1, 0x01F0F5, DUR_S },
    { 0x01F100, 0x01F10C, DUR_N },
    { 0x01F10D, 0x01F1AD, DUR_S },
    { 0x01F1E6, 0x01F202, DUR_S },
    { 0x01F210, 0x01F23B, DUR_S },
    { 0x01F240, 0x01F248, DUR_S },
    { 0x01F250, 0x01F251, DUR_S },
    { 0x01F260, 0x01F265, DUR_S },
    { 0x01F300, 0x01F6D7, DUR_S },
    { 0x01F6DC, 0x01F6EC, DUR_S },
    { 0x01F6F0, 0x01F6FC, DUR_S },
    { 0x01F700, 0x01F776, DUR_S },
    { 0x01F77B, 0x01F7D9, DUR_S },
    { 0x01F7E0, 0x01F7EB, DUR_S },
    { 0x01F7F0, 0x01F7F0, DUR_S },
    { 0x01F800, 0x01F80B, DUR_S },
    { 0x01F810, 0x01F847, DUR_S },
    { 0x01F850, 0x01F859, DUR_S },
    { 0x01F860, 0x01F887, DUR_S },
    { 0x01F890, 0x01F8AD, DUR_S },
    { 0x01F8B0, 0x01F8B1, DUR_S },
    { 0x01F900, 0x01FA53, DUR_S },
    { 0x01FA60, 0x01FA6D, DUR_S },
    { 0x01FA70, 0x01FA7C, DUR_S },
    { 0x01FA80, 0x01FA88, DUR_S },
    { 0x01FA90, 0x01FABD, DUR_S },
    { 0x01FABF, 0x01FAC5, DUR_S },
    { 0x01FACE, 0x01FADB, DUR_S },
    { 0x01FAE0, 0x01FAE8, DUR_S },
    { 0x01FAF0, 0x01FAF8, DUR_S },
    { 0x01FB00, 0x01FB92, DUR_S },
    { 0x01FB94, 0x01FBCA, DUR_S },
    { 0x01FBF0, 0x01FBF9, DUR_N },
    { 0x0E0100, 0x0E01EF, DUR_M },
};

static const size_t duration_cat_n = 879;

// Script ranges, byte-perfect copy of duration.py self.ranges.
// Stored as (end, script_index) pairs sorted by end ascending.
// bisect_left(end, cp) finds the range covering cp.

enum DurationScript {
    DUR_SCR_CJK = 0,
    DUR_SCR_HANGUL,
    DUR_SCR_KANA,
    DUR_SCR_ETHIOPIC,
    DUR_SCR_YI,
    DUR_SCR_INDIC,
    DUR_SCR_THAI_LAO,
    DUR_SCR_KHMER_MYANMAR,
    DUR_SCR_ARABIC,
    DUR_SCR_HEBREW,
    DUR_SCR_LATIN,
    DUR_SCR_CYRILLIC,
    DUR_SCR_GREEK,
    DUR_SCR_ARMENIAN,
    DUR_SCR_GEORGIAN,
    DUR_SCR_DEFAULT,
    // category-derived classes :
    DUR_SCR_PUNCTUATION,
    DUR_SCR_SPACE,
    DUR_SCR_DIGIT,
    DUR_SCR_MARK,
};

static const float duration_weights[] = {
    3.0f,  // CJK
    2.5f,  // HANGUL
    2.2f,  // KANA
    3.0f,  // ETHIOPIC
    3.0f,  // YI
    1.8f,  // INDIC
    1.5f,  // THAI_LAO
    1.8f,  // KHMER_MYANMAR
    1.5f,  // ARABIC
    1.5f,  // HEBREW
    1.0f,  // LATIN
    1.0f,  // CYRILLIC
    1.0f,  // GREEK
    1.0f,  // ARMENIAN
    1.0f,  // GEORGIAN
    1.0f,  // DEFAULT
    0.5f,  // PUNCTUATION
    0.2f,  // SPACE
    3.5f,  // DIGIT
    0.0f,  // MARK
};

struct DurationScriptRange {
    uint32_t end;
    uint8_t  script;
};

static const DurationScriptRange duration_script_ranges[88] = {
    { 0x02AF, DUR_SCR_LATIN         },
    { 0x03FF, DUR_SCR_GREEK         },
    { 0x052F, DUR_SCR_CYRILLIC      },
    { 0x058F, DUR_SCR_ARMENIAN      },
    { 0x05FF, DUR_SCR_HEBREW        },
    { 0x077F, DUR_SCR_ARABIC        },
    { 0x089F, DUR_SCR_ARABIC        },
    { 0x08FF, DUR_SCR_ARABIC        },
    { 0x097F, DUR_SCR_INDIC         },
    { 0x09FF, DUR_SCR_INDIC         },
    { 0x0A7F, DUR_SCR_INDIC         },
    { 0x0AFF, DUR_SCR_INDIC         },
    { 0x0B7F, DUR_SCR_INDIC         },
    { 0x0BFF, DUR_SCR_INDIC         },
    { 0x0C7F, DUR_SCR_INDIC         },
    { 0x0CFF, DUR_SCR_INDIC         },
    { 0x0D7F, DUR_SCR_INDIC         },
    { 0x0DFF, DUR_SCR_INDIC         },
    { 0x0EFF, DUR_SCR_THAI_LAO      },
    { 0x0FFF, DUR_SCR_INDIC         },
    { 0x109F, DUR_SCR_KHMER_MYANMAR },
    { 0x10FF, DUR_SCR_GEORGIAN      },
    { 0x11FF, DUR_SCR_HANGUL        },
    { 0x137F, DUR_SCR_ETHIOPIC      },
    { 0x139F, DUR_SCR_ETHIOPIC      },
    { 0x13FF, DUR_SCR_DEFAULT       },
    { 0x167F, DUR_SCR_DEFAULT       },
    { 0x169F, DUR_SCR_DEFAULT       },
    { 0x16FF, DUR_SCR_DEFAULT       },
    { 0x171F, DUR_SCR_DEFAULT       },
    { 0x173F, DUR_SCR_DEFAULT       },
    { 0x175F, DUR_SCR_DEFAULT       },
    { 0x177F, DUR_SCR_DEFAULT       },
    { 0x17FF, DUR_SCR_KHMER_MYANMAR },
    { 0x18AF, DUR_SCR_DEFAULT       },
    { 0x18FF, DUR_SCR_DEFAULT       },
    { 0x194F, DUR_SCR_INDIC         },
    { 0x19DF, DUR_SCR_INDIC         },
    { 0x19FF, DUR_SCR_KHMER_MYANMAR },
    { 0x1A1F, DUR_SCR_INDIC         },
    { 0x1AAF, DUR_SCR_INDIC         },
    { 0x1B7F, DUR_SCR_INDIC         },
    { 0x1BBF, DUR_SCR_INDIC         },
    { 0x1BFF, DUR_SCR_INDIC         },
    { 0x1C4F, DUR_SCR_INDIC         },
    { 0x1C7F, DUR_SCR_INDIC         },
    { 0x1C8F, DUR_SCR_CYRILLIC      },
    { 0x1CBF, DUR_SCR_GEORGIAN      },
    { 0x1CCF, DUR_SCR_INDIC         },
    { 0x1CFF, DUR_SCR_INDIC         },
    { 0x1D7F, DUR_SCR_LATIN         },
    { 0x1DBF, DUR_SCR_LATIN         },
    { 0x1DFF, DUR_SCR_DEFAULT       },
    { 0x1EFF, DUR_SCR_LATIN         },
    { 0x309F, DUR_SCR_KANA          },
    { 0x30FF, DUR_SCR_KANA          },
    { 0x312F, DUR_SCR_CJK           },
    { 0x318F, DUR_SCR_HANGUL        },
    { 0x9FFF, DUR_SCR_CJK           },
    { 0xA4CF, DUR_SCR_YI            },
    { 0xA4FF, DUR_SCR_DEFAULT       },
    { 0xA63F, DUR_SCR_DEFAULT       },
    { 0xA69F, DUR_SCR_CYRILLIC      },
    { 0xA6FF, DUR_SCR_DEFAULT       },
    { 0xA7FF, DUR_SCR_LATIN         },
    { 0xA82F, DUR_SCR_INDIC         },
    { 0xA87F, DUR_SCR_DEFAULT       },
    { 0xA8DF, DUR_SCR_INDIC         },
    { 0xA8FF, DUR_SCR_INDIC         },
    { 0xA92F, DUR_SCR_INDIC         },
    { 0xA95F, DUR_SCR_INDIC         },
    { 0xA97F, DUR_SCR_HANGUL        },
    { 0xA9DF, DUR_SCR_INDIC         },
    { 0xA9FF, DUR_SCR_KHMER_MYANMAR },
    { 0xAA5F, DUR_SCR_INDIC         },
    { 0xAA7F, DUR_SCR_KHMER_MYANMAR },
    { 0xAADF, DUR_SCR_INDIC         },
    { 0xAAFF, DUR_SCR_INDIC         },
    { 0xAB2F, DUR_SCR_ETHIOPIC      },
    { 0xAB6F, DUR_SCR_LATIN         },
    { 0xABBF, DUR_SCR_DEFAULT       },
    { 0xABFF, DUR_SCR_INDIC         },
    { 0xD7AF, DUR_SCR_HANGUL        },
    { 0xFAFF, DUR_SCR_CJK           },
    { 0xFDFF, DUR_SCR_ARABIC        },
    { 0xFE6F, DUR_SCR_DEFAULT       },
    { 0xFEFF, DUR_SCR_ARABIC        },
    { 0xFFEF, DUR_SCR_LATIN         },
};

static const size_t duration_script_n = 88;

// Decode one UTF-8 sequence at p, write the codepoint to *cp, return
// the byte count (0 on malformed).
static int duration_utf8_decode(const char * p, size_t avail, uint32_t * cp) {
    if (avail == 0) {
        return 0;
    }
    uint8_t c = (uint8_t) p[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && avail >= 2) {
        *cp = ((uint32_t) (c & 0x1F) << 6) | ((uint32_t) (uint8_t) p[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && avail >= 3) {
        *cp = ((uint32_t) (c & 0x0F) << 12) | ((uint32_t) ((uint8_t) p[1] & 0x3F) << 6) |
              ((uint32_t) (uint8_t) p[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && avail >= 4) {
        *cp = ((uint32_t) (c & 0x07) << 18) | ((uint32_t) ((uint8_t) p[1] & 0x3F) << 12) |
              ((uint32_t) ((uint8_t) p[2] & 0x3F) << 6) | ((uint32_t) (uint8_t) p[3] & 0x3F);
        return 4;
    }
    return 0;
}

// Binary search the category table for cp. Returns the class index
// (DUR_M / DUR_P / DUR_S / DUR_Z / DUR_N), or -1 if cp falls in a gap
// (i.e. its Unicode category is one of L / C / O which the estimator
// resolves through the script ranges instead).
static int duration_cat_lookup(uint32_t cp) {
    int lo = 0;
    int hi = (int) duration_cat_n - 1;
    while (lo <= hi) {
        int                      mid = (lo + hi) / 2;
        const DurationCatRange & r   = duration_cat_ranges[mid];
        if (cp < r.start) {
            hi = mid - 1;
        } else if (cp > r.end) {
            lo = mid + 1;
        } else {
            return (int) r.cls;
        }
    }
    return -1;
}

// Weight multiplier for a Latin long (macron) vowel relative to a plain letter.
#ifndef DUR_LONG_VOWEL_MULT
#    define DUR_LONG_VOWEL_MULT 2.0f
#endif

// Precomposed macron vowels: A E I O U Y (upper and lower)
static bool duration_is_macron_vowel(uint32_t cp) {
    switch (cp) {
        case 0x0100: case 0x0101:  // Ā ā
        case 0x0112: case 0x0113:  // Ē ē
        case 0x012A: case 0x012B:  // Ī ī
        case 0x014C: case 0x014D:  // Ō ō
        case 0x016A: case 0x016B:  // Ū ū
        case 0x0232: case 0x0233:  // Ȳ ȳ
            return true;
        default:
            return false;
    }
}

// Weight for one codepoint, mirrors RuleDurationEstimator._get_char_weight.
// Order of checks matches the Python reference exactly so the same
// codepoint resolves to the same weight on both sides.
static float duration_get_char_weight(uint32_t cp) {
    // ASCII letters take the latin fast path.
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) {
        return duration_weights[DUR_SCR_LATIN];
    }
    // ASCII space.
    if (cp == 0x20) {
        return duration_weights[DUR_SCR_SPACE];
    }
    // Arabic Tatweel: explicit pre-empt before the category check.
    if (cp == 0x0640) {
        return duration_weights[DUR_SCR_MARK];
    }

    // Classical Latin long vowels. A macron vowel lasts about twice a short
    // one, so it counts DUR_LONG_VOWEL_MULT times a plain Latin letter.
    // This is a deliberate divergence from upstream RuleDurationEstimator.
    //  - Precomposed (NFC) macron vowels return the full multiplied weight.
    //  - A combining macron U+0304 (NFD text) adds the extra on top of its
    //    base letter, which already counted 1.0 (the mark would otherwise be 0).
    if (duration_is_macron_vowel(cp)) {
        return duration_weights[DUR_SCR_LATIN] * DUR_LONG_VOWEL_MULT;
    }
    if (cp == 0x0304) {
        return duration_weights[DUR_SCR_LATIN] * (DUR_LONG_VOWEL_MULT - 1.0f);
    }

    // Unicode category check: M / P / S / Z / N short-circuit any
    // script-based weight.
    int cat = duration_cat_lookup(cp);
    if (cat >= 0) {
        switch (cat) {
            case DUR_M:
                return duration_weights[DUR_SCR_MARK];
            case DUR_P:
                return duration_weights[DUR_SCR_PUNCTUATION];
            case DUR_S:
                return duration_weights[DUR_SCR_PUNCTUATION];
            case DUR_Z:
                return duration_weights[DUR_SCR_SPACE];
            case DUR_N:
                return duration_weights[DUR_SCR_DIGIT];
        }
    }

    // bisect_left equivalent on script_ranges.end: finds the smallest
    // index i such that ranges[i].end >= cp.
    int lo = 0;
    int hi = (int) duration_script_n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (duration_script_ranges[mid].end < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < (int) duration_script_n) {
        return duration_weights[duration_script_ranges[lo].script];
    }

    // Upper planes: CJK Ext B/C/D and historic scripts default to CJK.
    if (cp > 0x20000) {
        return duration_weights[DUR_SCR_CJK];
    }
    return duration_weights[DUR_SCR_DEFAULT];
}

// Sum of per-character weights over a UTF-8 string.
static float duration_total_weight(const std::string & utf8) {
    float  total = 0.0f;
    size_t i     = 0;
    while (i < utf8.size()) {
        uint32_t cp = 0;
        int      n  = duration_utf8_decode(utf8.data() + i, utf8.size() - i, &cp);
        if (n == 0) {
            i++;
            continue;
        }
        total += duration_get_char_weight(cp);
        i += n;
    }
    return total;
}

// Estimate target duration in the same unit as ref_duration. Identical
// formula to RuleDurationEstimator.estimate_duration, including the
// power-curve boost when the linear estimate falls below low_threshold.
static float duration_estimate(const std::string & target_text,
                               const std::string & ref_text,
                               float               ref_duration,
                               float               low_threshold  = 50.0f,
                               float               boost_strength = 3.0f) {
    if (ref_duration <= 0.0f || ref_text.empty()) {
        return 0.0f;
    }
    float ref_weight = duration_total_weight(ref_text);
    if (ref_weight == 0.0f) {
        return 0.0f;
    }
    float speed_factor  = ref_weight / ref_duration;
    float target_weight = duration_total_weight(target_text);
    float estimated     = target_weight / speed_factor;
    if (estimated < low_threshold) {
        float alpha = 1.0f / boost_strength;
        return low_threshold * powf(estimated / low_threshold, alpha);
    }
    return estimated;
}

// Number of target audio tokens for a synthesis call. Mirrors
// OmniVoice._estimate_target_tokens: when no reference is available,
// falls back to the canonical 'Nice to meet you.' anchor at 25 tokens.
// Result is truncated to int (Python int() semantics) and floored to 1.
static int duration_estimate_tokens(const std::string & target_text,
                                    const std::string & ref_text,
                                    int                 ref_audio_tokens) {
    std::string rt;
    int         rd;
    if (ref_audio_tokens > 0 && !ref_text.empty()) {
        rt = ref_text;
        rd = ref_audio_tokens;
    } else {
        rt = "Nice to meet you.";
        rd = 25;
    }
    float est    = duration_estimate(target_text, rt, (float) rd);
    int   tokens = (int) est;
    if (tokens < 1) {
        tokens = 1;
    }
    return tokens;
}
