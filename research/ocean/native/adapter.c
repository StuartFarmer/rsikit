#include "g2048.h"

typedef struct {
    Game game;
    float action, reward, terminal;
} Slot;

Slot *ocean_create(int n, const uint32_t *seeds, float *observations) {
    Slot *slots = calloc(n, sizeof(Slot));
    if (!slots) return NULL;
    DictItem item = {.key = "scaffolding_ratio", .value = 0};
    Dict kwargs = {.items = &item, .size = 1, .cap = 1};
    for (int i = 0; i < n; i++) {
        Slot *s = &slots[i];
        s->game.agents[0].observations = observations + 16 * i;
        s->game.agents[0].actions = &s->action;
        s->game.agents[0].rewards = &s->reward;
        s->game.agents[0].terminals = &s->terminal;
        puf_init(&s->game, &kwargs);
        s->game.rng = seeds[i];
        puf_reset(&s->game);
    }
    return slots;
}

/* stats columns: merge score, max tile, shaped return, decisions, ending.
   endings: 0 active, 1 game over, 2 native timeout, 3 external cap. */
void ocean_step(Slot *slots, int count, const int64_t *active,
                const float *actions, int max_steps, float *rewards, double *stats) {
    for (int row = 0; row < count; row++) {
        int i = active[row];
        Slot *s = &slots[i];
        Game *g = &s->game;
        double *out = stats + 5 * i;
        if (out[4]) continue;
        Game before = *g;
        s->action = actions[row];
        puf_step(g);
        rewards[row] = s->reward;
        if (s->terminal) {
            /* puf_step already reset the live board. Read the completed log once.
               Replay only this last move on a copy with upstream's no-reset
               helper to distinguish game-over from timeout, including ties.
               Scratch buffers prevent this replay from changing observations. */
            float scratch_obs[16], scratch_terminal = 0;
            before.agents[0].observations = scratch_obs;
            before.agents[0].terminals = &scratch_terminal;
            step_without_reset(&before);
            out[0] = g->log.merge_score;
            out[1] = g->log.score;
            out[2] = g->log.episode_return;
            out[3] = g->log.episode_length;
            out[4] = scratch_terminal ? 1 : 2;
        } else {
            out[0] = g->score;
            out[1] = 1 << g->max_tile;
            out[2] = g->episode_reward;
            out[3] = g->tick;
            out[4] = g->tick >= max_steps ? 3 : 0;
            if (out[4] == 3) {
                /* Native max_tile predates spawning (or is zero after reset).
                   Inspect the capped board without changing native bookkeeping. */
                unsigned char max_tile = 0;
                for (int cell = 0; cell < 16; cell++) {
                    unsigned char tile = ((unsigned char *)g->grid)[cell];
                    if (tile > max_tile) max_tile = tile;
                }
                out[1] = 1 << max_tile;
            }
        }
    }
}

void ocean_close(Slot *slots, int n) {
    for (int i = 0; i < n; i++) puf_close(&slots[i].game);
    free(slots);
}
