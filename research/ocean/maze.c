/* Episode boundary for the pinned Ocean Maze functions extracted by install_maze.py.
 * Generate one map directly from its seed; never sample a shared map bank or autoreset.
 */
typedef struct {
    Env env;
    float action, reward, terminal;
} Episode;

void* maze_new(unsigned char* observation, int size) {
    Episode* episode = calloc(1, sizeof(Episode));
    if (!episode) return NULL;
    Env* env = &episode->env;
    env->num_agents = 1;
    env->state.width = env->state.height = size;
    env->agents[0].observations = observation;
    env->agents[0].actions = &episode->action;
    env->agents[0].rewards = &episode->reward;
    env->agents[0].terminals = &episode->terminal;
    return episode;
}

void maze_reset(Episode* episode, int seed, unsigned char* map) {
    Env* env = &episode->env;
    env->tick = 0;
    memset(&env->log, 0, sizeof(Log));
    episode->reward = episode->terminal = 0;
    create_maze_level(&env->state, 0.5f, seed);
    memcpy(map, env->state.maze, MAX_SIZE*MAX_SIZE);
    compute_observations(env);
}

int maze_step(Episode* episode, int action) {
    episode->action = action;
    puf_step(&episode->env);
    return episode->terminal != 0;
}

float maze_reward(Episode* episode) { return episode->reward; }
void maze_close(Episode* episode) { free(episode); }
