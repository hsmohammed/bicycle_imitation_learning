 # -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:58:40 2019

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
import rllab.misc.logger as logger
import time
from inverse_irl.csv2pickle import csv2pickle1
from inverse_irl.utils.general import TrainingIterator
import numpy as np

from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal, ProgBarCounter
from rllab.misc import tensor_utils
import itertools
from inverse_irl.utils.math_utils import gauss_log_pdf, categorical_log_pdf


tf.compat.v1.reset_default_graph()

expert_training = csv2pickle1()

env = TfEnv(GymEnv('bicycle-v0', record_video=False, record_log=False))
experts = expert_training
#    experts1 = load_latest_experts('data/bicycle', n=5)
tf.compat.v1.disable_eager_execution()

irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(64, 64))
algo = IRLTRPO(
    env=env,
    policy=policy,
    irl_model=irl_model,
    n_itr=100,
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

expert_obs, expert_acts = algo.irl_model.extract_paths(algo.irl_model.expert_trajs, keys=('observations', 'actions'))



    
    
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('out/out.meta')
    saver.restore(sess, "out/out")