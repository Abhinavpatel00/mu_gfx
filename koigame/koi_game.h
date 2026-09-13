#ifndef KOI_GAME_H
#define KOI_GAME_H

#include <stdint.h>
#include "koi_pond_host.h"

enum {
    KOI_FOOD_COUNT = 8,
};

typedef struct KoiGameVec2 {
    float x;
    float y;
} KoiGameVec2;

typedef struct KoiFish {
    KoiGameVec2 position;
    KoiGameVec2 velocity;
    KoiGameVec2 target;

    float size;
    float heading;
    float wag;
    float cruise;

    float hunger;
    float fear;
    float happiness;

    uint32_t type;
    float seed;
} KoiFish;

typedef struct KoiRipple {
    KoiGameVec2 position;
    float birth_time;
    float amplitude;
} KoiRipple;

typedef struct KoiFood {
    KoiGameVec2 position;
    float life;
    float value;
    uint32_t active;
} KoiFood;

typedef enum KoiInputButton {
    KOI_INPUT_NONE  = 0,
    KOI_INPUT_FEED  = 1,
    KOI_INPUT_STIR  = 2,
} KoiInputButton;

typedef struct KoiGame {
    KoiFish fish[KOI_FISH_COUNT];
    KoiRipple ripples[KOI_RIPPLE_COUNT];
    KoiFood food[KOI_FOOD_COUNT];

    uint32_t ripple_cursor;
    uint32_t rng_state;
    uint32_t score;
    uint32_t food_stock;
    uint32_t eaten_count;

    float time;
    float panic;
    float round_time;
    float combo_time;
    uint32_t combo;
} KoiGame;

void koi_game_init(KoiGame *game, uint32_t seed);
void koi_game_update(KoiGame *game, float dt);
void koi_game_input(KoiGame *game, KoiInputButton button, KoiGameVec2 pond_position);
void koi_game_write_gpu(const KoiGame *game, KoiPondData *gpu);

#endif
