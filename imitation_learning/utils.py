from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from .buffer import Buffer

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_multiple_demo(envs, algos, buffer_sizes, device, std, p_rand, seed=0, interval=1, env_ids=None):
    env = envs[0]
    if env_ids is None:
        buffer = Buffer(
            buffer_size=buffer_sizes,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device
        )
    else:
        buffer = Buffer(
            buffer_size=buffer_sizes,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            code_shape=len(env_ids[0])
        )


    for i in range(len(envs)):
        env = envs[i]
        algo = algos[i]
        buffer_size = int(buffer_sizes / len(envs))
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        total_return = 0.0
        num_episodes = 0

        state = env.reset()
        t = 0
        episode_return = 0.0
        all_steps = 0

        last_state = None
        last_action = None

        for _ in tqdm(range(1, (buffer_size + 1) * interval)):
            t += 1
            all_steps += 1
            if np.random.rand() < p_rand:
                action = env.action_space.sample()
            else:
                action = algo.exploit(state)
                action = add_random_noise(action, std)

            next_state, reward, done, _ = env.step(action)
            mask = False if t == env._max_episode_steps else done

            if all_steps % interval == 0 and last_state is not None:
                next_action = add_random_noise(algo.exploit(state), std)
                if env_ids is not None:
                    buffer.append(state, action, reward, mask, last_state, last_action, next_state, next_action, np.array(env_ids[i]))
                else:
                    buffer.append(state, action, reward, mask, last_state, last_action, next_state, next_action)

            episode_return += reward

            if done:
                num_episodes += 1
                total_return += episode_return
                state = env.reset()
                t = 0
                episode_return = 0.0
                last_state = None
                last_action = None

            else:
                last_state = state
                last_action = action
                state = next_state

        print(f'Mean return of the expert is {total_return / num_episodes}')

    return buffer


def collect_multiple_demo_transform(envs, algos, buffer_sizes, device, std, p_rand, seed=0, interval=1, env_ids=None):
    env = envs[0]
    if env_ids == None:
        buffer = Buffer(
            buffer_size=buffer_sizes,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device
        )
    else:
        buffer = Buffer(
            buffer_size=buffer_sizes,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            code_shape=len(env_ids[0])
        )


    for i in range(len(envs)):
        env = envs[i]
        algo = algos[i]
        buffer_size = int(buffer_sizes / len(envs))
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        total_return = 0.0
        num_episodes = 0

        state = env.reset()
        t = 0
        episode_return = 0.0
        all_steps = 0

        last_state = None
        last_action = None

        for _ in tqdm(range(1, (buffer_size + 1) * interval)):
            t += 1
            all_steps += 1
            raw_state = env.get_raw_obs()
            if np.random.rand() < p_rand:
                action = env.action_space.sample()
            else:
                action = algo.exploit(raw_state)
                action = add_random_noise(action, std)

            next_state, reward, done, _ = env.step(action)
            mask = False if t == env._max_episode_steps else done

            if all_steps % interval == 0 and last_state is not None:
                next_action = add_random_noise(algo.exploit(state), std)
                if env_ids is not None:
                    buffer.append(state, action, reward, mask, last_state, last_action, next_state, next_action, np.array(env_ids[i]))
                else:
                    buffer.append(state, action, reward, mask, last_state, last_action, next_state, next_action)

            episode_return += reward

            if done:
                num_episodes += 1
                total_return += episode_return
                state = env.reset()
                t = 0
                episode_return = 0.0
                last_state = None
                last_action = None

            else:
                last_state = state
                last_action = action
                state = next_state

        print(f'Mean return of the expert is {total_return / num_episodes}')

    return buffer

def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0, interval=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    all_steps = 0

    last_state = None
    last_action = None

    for _ in tqdm(range(1, (buffer_size + 1) * interval)):
        t += 1
        all_steps += 1
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        if all_steps % interval == 0 and last_state is not None:
            next_action = add_random_noise(algo.exploit(state), std)
            buffer.append(state, action, reward, mask, last_state, last_action, next_state, next_action)

        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            last_state = None
            last_action = None

        else:
            last_state = state
            last_action = action
            state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer



def collect_demo_transform(env, algo, buffer_size, device, std, p_rand, seed=0, interval=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    all_steps = 0

    last_state = None
    last_action = None

    for _ in tqdm(range(1, (buffer_size + 1) * interval)):
        t += 1
        all_steps += 1
        raw_state = env.get_raw_obs()
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(raw_state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        if all_steps % interval == 0 and last_state is not None:
            next_action = add_random_noise(algo.exploit(state), std)
            buffer.append(state, action, reward, mask, last_state, last_action, next_state, next_action)

        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            last_state = None
            last_action = None

        else:
            last_state = state
            last_action = action
            state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer
