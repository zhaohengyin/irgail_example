import os
import argparse
from datetime import datetime
import torch
from custom_env.idp2.register import register_inverted_double_pendulum
from imitation_learning.env import make_env
from imitation_learning.buffer import SerializedBuffer
from imitation_learning.algo.cdil import CDIL
from imitation_learning.trainer import Trainer
from repr.representation import RepresentationNetwork


def run(args):
    env_id = register_inverted_double_pendulum([args.c1, args.c2, args.c3])
    env = make_env(env_id)
    env_test = make_env(env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    embedding = RepresentationNetwork(env.observation_space.shape[0], env.action_space.shape[0],
                              3, 12, 8, None, h_forward=(), h_sinv=(128,))

    embedding.load(args.embedding)
    algo = CDIL(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        encoder=embedding,
        zs_dim=[12],
        za_dim=[8],
        pi_code=[args.c1, args.c2, args.c3],
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        mode=args.mode
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.eval_n,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True, default='./assets/idp_expert_buffer.pth')
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--embedding', type=str, default='./assets/idp_pretrained.pth')
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--eval_n', type=int, default=5)
    p.add_argument('--env_id', type=str, default='InvertedDoublePendulum-v2')
    p.add_argument('--algo', type=str, default='CDIL')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mode', type=str, default='n')
    p.add_argument('--c1', type=float, default=0.6)
    p.add_argument('--c2', type=float, default=0.6)
    p.add_argument('--c3', type=float, default=1.0)
    args = p.parse_args()
    run(args)
