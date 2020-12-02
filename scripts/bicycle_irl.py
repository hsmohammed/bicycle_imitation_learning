# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 20:10:17 2019

@author: Office
"""

import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_irl.algos.irl_trpo import IRLTRPO
from inverse_irl.models.imitation_learning import AIRLStateAction, GAIL, GAN_GCL
from inverse_irl.utils.log_utils import rllab_logdir, load_latest_experts
import gym_bicycle

def main():
    env = TfEnv(GymEnv('bicycle-v0', record_video=False, record_log=False))
    
    experts = expert_training
#    experts1 = load_latest_experts('data/bicycle', n=5)

    irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(64, 64))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=5000,
        batch_size=1000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/bicycle_AIRL3'):
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    main()