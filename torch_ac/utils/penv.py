import multiprocessing
import gymnasium as gym


multiprocessing.set_start_method("fork")

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, agent_pos, agent_dir, info = env.step(data)
            if terminated or truncated:
                obs, _ = env.reset()
            conn.send((obs, reward, terminated, truncated, agent_pos, agent_dir, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    # # mo gai ban
    # def __init__(self, envs):
    #     assert len(envs) >= 1, "No environment given."

    #     self.envs = envs
    #     self.observation_space = self.envs[0].observation_space
    #     self.action_space = self.envs[0].action_space

    #     self.locals = []
    #     for env in self.envs:
    #         local, remote = multiprocessing.Pipe()
    #         self.locals.append(local)
    #         p = multiprocessing.Process(target=worker, args=(remote, env))
    #         p.daemon = True
    #         p.start()
    #         remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        actions, actions_scale = actions
        for local, action, action_scale in zip(self.locals, actions[1:], actions_scale[1:]):
            local.send(("step", (action, action_scale)))
        obs, reward, terminated, truncated, agent_pos, agent_dir, info = self.envs[0].step((actions[0], actions_scale[0]))
        if terminated or truncated:
            obs, _ = self.envs[0].reset()
        results = zip(*[(obs, reward, terminated, truncated, agent_pos, agent_dir, info)] + [local.recv() for local in self.locals])        
        return results

    # # mo gai ban
    # def step(self, actions):
    #     actions, actions_scale = actions
    #     for local, action, action_scale in zip(self.locals, actions, actions_scale):
    #         local.send(("step", (action, action_scale)))
    #     # obs, reward, terminated, truncated, info = self.envs[0].step((actions[0], actions_scale[0]))
    #     # if terminated or truncated:
    #     #     obs, _ = self.envs[0].reset()
    #     results = zip(*[local.recv() for local in self.locals])
    #     return results

    def render(self):
        raise NotImplementedError
