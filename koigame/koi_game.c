#include "koi_game.h"

#include <math.h>
#include <stddef.h>

#define KOI_PI 3.14159265358979323846f
#define KOI_TAU 6.28318530717958647692f

static const float k_pad_x[19] = {
    0.10f,0.30f,0.02f,0.24f,0.06f,0.20f,0.15f,0.45f,0.75f,
    2.62f,2.88f,2.96f,2.45f,2.94f,2.80f,2.50f,2.06f,2.24f,1.72f
};

static const float k_pad_y[19] = {
    0.940f,0.965f,0.760f,0.800f,0.420f,0.280f,0.070f,0.045f,0.040f,
    0.945f,0.895f,0.700f,0.830f,0.450f,0.095f,0.045f,0.600f,0.360f,0.800f
};

static const float k_pad_radius[19] = {
    0.150f,0.115f,0.115f,0.085f,0.125f,0.095f,0.145f,0.105f,0.085f,
    0.160f,0.125f,0.100f,0.080f,0.115f,0.135f,0.090f,0.095f,0.070f,0.060f
};

static float
koi_rand01(KoiGame *game)
{
    // xorshift32 keeps gameplay deterministic and avoids libc rand state.
    uint32_t x = game->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    game->rng_state = x ? x : 0x6d2b79f5u;

    return (float)(game->rng_state & 0x00ffffffu) / 16777215.0f;
}

static KoiGameVec2
koi_add(KoiGameVec2 a, KoiGameVec2 b)
{
    return (KoiGameVec2){a.x + b.x, a.y + b.y};
}

static KoiGameVec2
koi_sub(KoiGameVec2 a, KoiGameVec2 b)
{
    return (KoiGameVec2){a.x - b.x, a.y - b.y};
}

static KoiGameVec2
koi_mul(KoiGameVec2 a, float s)
{
    return (KoiGameVec2){a.x * s, a.y * s};
}

static float
koi_length(KoiGameVec2 v)
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

static KoiGameVec2
koi_normalize(KoiGameVec2 v)
{
    float length = koi_length(v);
    if (length <= 1e-6f)
        return (KoiGameVec2){0.0f, 0.0f};

    return koi_mul(v, 1.0f / length);
}

static int
koi_is_valid_target(KoiGameVec2 p, float fish_size)
{
    for (size_t i = 0; i < 19; ++i) {
        float dx = p.x - k_pad_x[i];
        float dy = p.y - k_pad_y[i];
        float radius = k_pad_radius[i] + 0.16f + fish_size;

        if (dx * dx + dy * dy < radius * radius)
            return 0;
    }

    return p.x >= 0.45f && p.x <= 2.55f &&
           p.y >= 0.20f && p.y <= 0.80f;
}

static void
koi_set_target(KoiGame *game, KoiFish *fish)
{
    for (uint32_t attempt = 0; attempt < 12; ++attempt) {
        KoiGameVec2 target = {
            0.50f + koi_rand01(game) * 2.00f,
            0.22f + koi_rand01(game) * 0.56f,
        };

        if (koi_is_valid_target(target, fish->size)) {
            fish->target = target;
            return;
        }
    }

    // Fallback is deterministic and always inside the preferred swimming box.
    fish->target = (KoiGameVec2){1.5f, 0.5f};
}

static void
koi_push_ripple(KoiGame *game, KoiGameVec2 position, float amplitude)
{
    uint32_t index = game->ripple_cursor;

    game->ripples[index].position = position;
    game->ripples[index].birth_time = game->time;
    game->ripples[index].amplitude = amplitude;

    game->ripple_cursor = (index + 1u) % KOI_RIPPLE_COUNT;
}

static void
koi_stir(KoiGame *game, KoiGameVec2 position)
{
    koi_push_ripple(game, position, 0.0038f);

    for (uint32_t i = 0; i < KOI_FISH_COUNT; ++i) {
        KoiFish *fish = &game->fish[i];
        KoiGameVec2 delta = koi_sub(fish->position, position);
        float distance = koi_length(delta);

        if (distance <= 1e-5f || distance >= 0.45f)
            continue;

        float q = 1.0f - distance / 0.45f;
        float impulse = 0.55f * q * q;

        fish->velocity = koi_add(
            fish->velocity,
            koi_mul(koi_normalize(delta), impulse));

        fish->fear = fminf(1.0f, fish->fear + impulse * 0.8f);
    }

    game->panic = fminf(1.0f, game->panic + 0.10f);
}

static void
koi_feed(KoiGame *game, KoiGameVec2 position)
{
    if (game->food_stock == 0)
        return;

    for (uint32_t i = 0; i < KOI_FOOD_COUNT; ++i) {
        KoiFood *food = &game->food[i];

        if (food->active)
            continue;

        food->position = position;
        food->life = 7.0f;
        food->value = 1.0f;
        food->active = 1;

        --game->food_stock;
        koi_push_ripple(game, position, 0.0028f);
        return;
    }
}

static void
koi_update_food(KoiGame *game, float dt)
{
    for (uint32_t i = 0; i < KOI_FOOD_COUNT; ++i) {
        KoiFood *food = &game->food[i];

        if (!food->active)
            continue;

        food->life -= dt;
        if (food->life <= 0.0f) {
            food->active = 0;
            continue;
        }

        for (uint32_t j = 0; j < KOI_FISH_COUNT; ++j) {
            KoiFish *fish = &game->fish[j];
            float distance = koi_length(
                koi_sub(fish->position, food->position));

            if (distance >= 0.07f)
                continue;

            food->active = 0;
            fish->hunger = fmaxf(0.0f, fish->hunger - food->value);
            fish->happiness = fminf(1.0f, fish->happiness + 0.20f);

            if (game->combo_time > 0.0f)
                ++game->combo;
            else
                game->combo = 1;

            game->combo_time = 2.0f;
            game->score += 100u * game->combo;
            ++game->eaten_count;
            break;
        }
    }
}

static void
koi_update_fish(KoiGame *game, float dt)
{
    for (uint32_t i = 0; i < KOI_FISH_COUNT; ++i) {
        KoiFish *fish = &game->fish[i];

        KoiGameVec2 to_target = koi_sub(fish->target, fish->position);
        float target_distance = koi_length(to_target);

        if (target_distance < 0.12f)
            koi_set_target(game, fish);

        KoiGameVec2 desired = koi_mul(
            koi_normalize(to_target), fish->cruise);

        for (uint32_t f = 0; f < KOI_FOOD_COUNT; ++f) {
            const KoiFood *food = &game->food[f];
            if (!food->active)
                continue;

            KoiGameVec2 delta = koi_sub(food->position, fish->position);
            float distance = koi_length(delta);

            if (distance < 0.75f) {
                float weight = 1.0f - distance / 0.75f;
                float attract = 0.10f * weight * (0.35f + fish->hunger);
                desired = koi_add(
                    desired,
                    koi_mul(koi_normalize(delta), attract));
            }
        }

        for (size_t p = 0; p < 19; ++p) {
            float dx = fish->position.x - k_pad_x[p];
            float dy = fish->position.y - k_pad_y[p];
            float distance = sqrtf(dx * dx + dy * dy);
            float radius = k_pad_radius[p] + 0.10f + fish->size * 0.5f;

            if (distance > 1e-5f && distance < radius + 0.15f) {
                float push = (radius + 0.15f - distance) / 0.15f * 0.25f;
                desired.x += dx / distance * push;
                desired.y += dy / distance * push;
            }
        }

        if (fish->position.x < 0.45f)
            desired.x += (0.45f - fish->position.x) * 0.8f;
        if (fish->position.x > 2.55f)
            desired.x -= (fish->position.x - 2.55f) * 0.8f;
        if (fish->position.y < 0.20f)
            desired.y += (0.20f - fish->position.y) * 0.8f;
        if (fish->position.y > 0.80f)
            desired.y -= (fish->position.y - 0.80f) * 0.8f;

        for (uint32_t j = 0; j < KOI_FISH_COUNT; ++j) {
            if (i == j)
                continue;

            KoiGameVec2 delta = koi_sub(fish->position, game->fish[j].position);
            float distance = koi_length(delta);

            if (distance > 1e-5f && distance < 0.30f) {
                float push = (0.30f - distance) * 0.5f;
                desired = koi_add(
                    desired,
                    koi_mul(koi_normalize(delta), push));
            }
        }

        // Fear reduces steering coherence instead of teleporting or freezing.
        desired = koi_mul(desired, 1.0f - 0.35f * fish->fear);

        float response = fminf(1.0f, dt * 1.6f);
        fish->velocity.x += (desired.x - fish->velocity.x) * response;
        fish->velocity.y += (desired.y - fish->velocity.y) * response;

        float speed = koi_length(fish->velocity);
        if (speed > 0.55f) {
            fish->velocity = koi_mul(fish->velocity, 0.55f / speed);
            speed = 0.55f;
        }

        fish->position = koi_add(
            fish->position,
            koi_mul(fish->velocity, dt));

        if (speed > 0.015f) {
            float target_heading = atan2f(
                fish->velocity.y,
                fish->velocity.x);

            float delta = target_heading - fish->heading;
            while (delta > KOI_PI)  delta -= KOI_TAU;
            while (delta < -KOI_PI) delta += KOI_TAU;

            fish->heading += delta * fminf(1.0f, dt * 2.5f);
        }

        fish->wag += dt * (2.2f + speed * 24.0f);
        fish->hunger = fminf(1.0f, fish->hunger + dt * 0.01f);
        fish->fear = fmaxf(0.0f, fish->fear - dt * 0.7f);
        fish->happiness = fmaxf(0.0f, fish->happiness - dt * 0.015f);

        if (speed > 0.12f && fish->fear < 0.5f && koi_rand01(game) < dt * 0.9f)
            koi_push_ripple(game, fish->position, 0.0007f);
    }
}

void
koi_game_init(KoiGame *game, uint32_t seed)
{
    *game = (KoiGame){0};
    game->rng_state = seed ? seed : 0x12345678u;
    game->food_stock = 20;
    game->round_time = 60.0f;
    game->combo = 1;

    static const float initial_x[KOI_FISH_COUNT] = {
        0.85f, 2.15f, 1.85f, 1.35f, 0.75f, 2.50f
    };
    static const float initial_y[KOI_FISH_COUNT] = {
        0.62f, 0.68f, 0.42f, 0.30f, 0.22f, 0.18f
    };
    static const float initial_size[KOI_FISH_COUNT] = {
        0.105f, 0.098f, 0.112f, 0.100f, 0.093f, 0.088f
    };
    static const uint32_t initial_type[KOI_FISH_COUNT] = {0,1,4,2,3,5};
    static const float initial_seed[KOI_FISH_COUNT] = {0.31f,0.77f,0.12f,0.55f,0.90f,0.44f};

    for (uint32_t i = 0; i < KOI_FISH_COUNT; ++i) {
        KoiFish *fish = &game->fish[i];
        fish->position = (KoiGameVec2){initial_x[i], initial_y[i]};
        fish->velocity = (KoiGameVec2){0.03f, 0.0f};
        fish->size = initial_size[i];
        fish->heading = koi_rand01(game) * KOI_TAU;
        fish->wag = koi_rand01(game) * 10.0f;
        fish->cruise = 0.05f + koi_rand01(game) * 0.035f;
        fish->hunger = 0.35f;
        fish->fear = 0.0f;
        fish->happiness = 0.5f;
        fish->type = initial_type[i];
        fish->seed = initial_seed[i];
        koi_set_target(game, fish);
    }
}

void
koi_game_input(KoiGame *game, KoiInputButton button, KoiGameVec2 pond_position)
{
    if (button == KOI_INPUT_FEED)
        koi_feed(game, pond_position);
    else if (button == KOI_INPUT_STIR)
        koi_stir(game, pond_position);
}

void
koi_game_update(KoiGame *game, float dt)
{
    game->time += dt;
    game->round_time = fmaxf(0.0f, game->round_time - dt);
    game->combo_time = fmaxf(0.0f, game->combo_time - dt);

    if (game->combo_time <= 0.0f)
        game->combo = 1;

    // Panic decays only when the player stops being a menace to aquatic life.
    game->panic = fmaxf(0.0f, game->panic - dt * 0.035f);

    koi_update_food(game, dt);
    koi_update_fish(game, dt);
}

void
koi_game_write_gpu(const KoiGame *game, KoiPondData *gpu)
{
    for (uint32_t i = 0; i < KOI_FISH_COUNT; ++i) {
        const KoiFish *fish = &game->fish[i];

        gpu->fish[i].x = fish->position.x;
        gpu->fish[i].y = fish->position.y;
        gpu->fish[i].z = fish->heading;
        gpu->fish[i].w = fish->wag;

        gpu->fish_meta[i].x = fish->size;
        gpu->fish_meta[i].y = (float)fish->type;
        gpu->fish_meta[i].z = fish->seed;
        gpu->fish_meta[i].w = 0.0f;
    }

    for (uint32_t i = 0; i < KOI_RIPPLE_COUNT; ++i) {
        const KoiRipple *ripple = &game->ripples[i];

        gpu->ripple[i].x = ripple->position.x;
        gpu->ripple[i].y = ripple->position.y;
        gpu->ripple[i].z = ripple->birth_time;
        gpu->ripple[i].w = ripple->amplitude;
    }
}
