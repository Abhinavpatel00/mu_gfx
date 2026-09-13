#ifndef KOI_POND_HOST_H
#define KOI_POND_HOST_H

#include <stdint.h>

// Keep every GPU-facing field float4-sized. This avoids surprise padding
// differences when the same conceptual data is consumed by C and Slang.

enum {
    KOI_FISH_COUNT   = 6,
    KOI_RIPPLE_COUNT = 24,
    KOI_FOOD_COUNT   = 8,
};

typedef struct KoiPondGPU {
    vec4 fish[KOI_FISH_COUNT];       /* xy=pos  z=heading  w=wag phase   */
    vec4 fish_meta[KOI_FISH_COUNT];  /* x=size y=type z=seed  w=mood     */
    vec4 ripple[KOI_RIPPLE_COUNT];   /* xy=pos  z=birth    w=amplitude   */
    vec4 food[KOI_FOOD_COUNT];       /* xy=pos  z=birth    w=active      */
    vec4 predator;                   /* xy=pos  z=state    w=timer       */
    vec4 game_misc;                  /* x=score y=combo z=panic w=stock  */
} KoiPondGPU;

_Static_assert(sizeof(KoiPondGPU) % 16 == 0, "layout drifted");
_Static_assert(sizeof(KoiPondVec4) == 16, "KoiPondVec4 must be 16 bytes");
_Static_assert(sizeof(KoiPondData) == 16 * (6 + 6 + 24),
               "KoiPondData layout drifted");

#endif
