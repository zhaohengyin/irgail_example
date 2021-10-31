from repr.representation import RepresentationNetwork
from repr.buffer import Buffer
import gym
from datetime import datetime
import os
import argparse

p = argparse.ArgumentParser()
p.add_argument('--env', type=str, default='InvertedDoublePendulum-v2')
p.add_argument('--buffer', type=str, default='./assets/idp_expert_random_buffer.pkl')
p.add_argument('--c_f', type=float, default=5.0)
args = p.parse_args()

time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    './representation_logs/' + args.env, str(time))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make(args.env)
buffer = Buffer(args.buffer)

algo = RepresentationNetwork(s_dim=env.observation_space.shape[0],
                             a_dim=env.action_space.shape[0],
                             c_dim=3,
                             zs_dim=12,
                             za_dim=8,
                             buffer=buffer,
                             h_forward=(),
                             h_sinv=(128,),
                             c_f=args.c_f,
                             name='idp')

algo.train(20, 500, log_dir)

