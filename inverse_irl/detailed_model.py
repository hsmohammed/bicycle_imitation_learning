# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:44:20 2019

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



expert_training = csv2pickle1()

env = TfEnv(GymEnv('bicycle-v0', record_video=False, record_log=False))

experts = expert_training
#    experts1 = load_latest_experts('data/bicycle', n=5)

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

with rllab_logdir(algo=algo, dirname='data/bicycle_AIRL3'):
    with tf.Session():
#        algo.train()
        
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        if algo.init_pol_params is not None:
            algo.policy.set_param_values(algo.init_pol_params)
        if algo.init_irl_params is not None:
            algo.irl_model.set_params(algo.init_irl_params)
        algo.start_worker()
        
        start_time = time.time()

        returns = []
        for itr in range(algo.start_itr, algo.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                
                
                
#                paths = algo.obtain_samples(itr)
                
                # Obtaining Samples
                
                logger.log("Obtaining samples for iteration %d..." % itr)
                paths = []
                n_samples = 0
                obses = algo.sampler.vec_env.reset()
                dones = np.asarray([True] * algo.sampler.vec_env.num_envs)
                running_paths = [None] * algo.sampler.vec_env.num_envs
        
                pbar = ProgBarCounter(algo.sampler.algo.batch_size)
                policy_time = 0
                env_time = 0
                process_time = 0
        
                policy = algo.sampler.algo.policy
                import time
                while n_samples < algo.sampler.algo.batch_size:
                    t = time.time()
                    policy.reset(dones)
                    actions, agent_infos = policy.get_actions(obses)
        
                    policy_time += time.time() - t
                    t = time.time()
                    next_obses, rewards, dones, env_infos = algo.sampler.vec_env.step(actions)
                    env_time += time.time() - t
        
                    t = time.time()
        
                    agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
                    env_infos = tensor_utils.split_tensor_dict_list(env_infos)
                    if env_infos is None:
                        env_infos = [dict() for _ in range(algo.sampler.vec_env.num_envs)]
                    if agent_infos is None:
                        agent_infos = [dict() for _ in range(algo.sampler.vec_env.num_envs)]
                    for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                            rewards, env_infos, agent_infos,
                                                                                            dones):
                        if running_paths[idx] is None:
                            running_paths[idx] = dict(
                                observations=[],
                                actions=[],
                                rewards=[],
                                env_infos=[],
                                agent_infos=[],
                            )
                        running_paths[idx]["observations"].append(observation)
                        running_paths[idx]["actions"].append(action)
                        running_paths[idx]["rewards"].append(reward)
                        running_paths[idx]["env_infos"].append(env_info)
                        running_paths[idx]["agent_infos"].append(agent_info)
                        if done:
                            paths.append(dict(
                                observations=algo.sampler.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                                actions=algo.sampler.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                                rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                                env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                                agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                            ))
                            n_samples += len(running_paths[idx]["rewards"])
                            running_paths[idx] = None
                    process_time += time.time() - t
                    pbar.inc(len(obses))
                    obses = next_obses
        
                pbar.stop()
        
                logger.record_tabular("PolicyExecTime", policy_time)
                logger.record_tabular("EnvExecTime", env_time)
                logger.record_tabular("ProcessExecTime", process_time)
                #############################
                #############################
                #############################
                ## End obtaining samples
                



                logger.log("Processing samples...")
                
                
#                paths = algo.compute_irl(paths, itr=itr)
                
                if algo.no_reward:
                    tot_rew = 0
                    for path in paths:
                        tot_rew += np.sum(path['rewards'])
                        path['rewards'] *= 0
                    logger.record_tabular('OriginalTaskAverageReturn', tot_rew/float(len(paths)))


                if algo.train_irl:
                    max_itrs = algo.discrim_train_itrs
                    lr=1e-3
                    
#                    mean_loss = algo.irl_model.fit(paths, policy=algo.policy, itr=itr, max_itrs=max_itrs, lr=lr,
#                                                   logger=logger)
                    
                    batch_size=32
                    
                    algo.irl_model.eval_expert_probs(paths, policy, insert=True)
                    
                    # Evaluate Expert Trajectories
                    
                    
                    for traj in experts:
                        if 'agent_infos' in traj:
                            del traj['agent_infos']
                        if 'a_logprobs' in traj:
                            del traj['a_logprobs']
            
                    if isinstance(policy, np.ndarray):
                        algo.irl_model._compute_path_probs(experts, insert=insert)
                    elif hasattr(policy, 'recurrent') and policy.recurrent:
                        policy.reset([True]*len(experts))
                        expert_obs = algo.irl_model.extract_paths(experts, keys=('observations',), stack=True)[0]
                        agent_infos = []
                        for t in range(expert_obs.shape[1]):
                            a, infos = policy.get_actions(expert_obs[:, t])
                            agent_infos.append(infos)
                        agent_infos_stack = tensor_utils.stack_tensor_dict_list(agent_infos)
                        for key in agent_infos_stack:
                            agent_infos_stack[key] = np.transpose(agent_infos_stack[key], axes=[1,0,2])
                        agent_infos_transpose = tensor_utils.split_tensor_dict_list(agent_infos_stack)
                        for i, path in enumerate(experts):
                            path['agent_infos'] = agent_infos_transpose[i]
                    else:
                        for path in experts:
                            actions, agent_infos = policy.get_actions(path['observations'])
                            path['agent_infos'] = agent_infos
                            
#                    algo.irl_model._compute_path_probs(experts, insert=True)
                    
                    DIST_GAUSSIAN = 'gaussian'
                    DIST_CATEGORICAL = 'categorical'
                    insert = True
                    """
                    Returns a N x T matrix of action probabilities
                    """
                    insert_key='a_logprobs'
                    pol_dist_type = None
                    
                    if insert_key in experts[0]:
                        np.array([path[insert_key] for path in experts])
            
                    if pol_dist_type is None:
                        # try to  infer distribution type
                        path0 = experts[0]
                        if 'log_std' in path0['agent_infos']:
                            pol_dist_type = DIST_GAUSSIAN
                        elif 'prob' in path0['agent_infos']:
                            pol_dist_type = DIST_CATEGORICAL
                        else:
                            raise NotImplementedError()
                            
                    # compute path probs
                    Npath = len(experts)
                    actions = [path['actions'] for path in experts]
                    if pol_dist_type == DIST_GAUSSIAN:
                        params = [(path['agent_infos']['mean'], path['agent_infos']['log_std']) for path in experts]
                        path_probs = [gauss_log_pdf(params[i], actions[i]) for i in range(Npath)]
                    elif pol_dist_type == DIST_CATEGORICAL:
                        params = [(path['agent_infos']['prob'],) for path in experts]
                        path_probs = [categorical_log_pdf(params[i], actions[i]) for i in range(Npath)]
                    else:
                        raise NotImplementedError("Unknown distribution type")
            
                    if insert:
                        for i, path in enumerate(experts):
                            path[insert_key] = path_probs[i]
            
                    np.array(path_probs)
                    
                    
                    
                    
#                    algo.irl_model.eval_expert_probs(algo.irl_model.expert_trajs, policy, insert=True)
                    obs, acts, path_probs = algo.irl_model.extract_paths(paths, keys=('observations', 'actions', 'a_logprobs'))
                    expert_obs, expert_acts, expert_probs = algo.irl_model.extract_paths(algo.irl_model.expert_trajs, keys=('observations', 'actions', 'a_logprobs'))

                        # Train discriminator
                    for it in TrainingIterator(max_itrs, heartbeat=5):
                        obs_batch, act_batch, lprobs_batch = \
                            algo.irl_model.sample_batch(obs, acts, path_probs, batch_size=batch_size)
            
                        expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                            algo.irl_model.sample_batch(expert_obs, expert_acts, expert_probs, batch_size=batch_size)
            
                        labels = np.zeros((batch_size*2, 1))
                        labels[batch_size:] = 1.0
                        obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
                        act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
                        lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)
            
                        loss, _ = tf.get_default_session().run([algo.irl_model.loss, algo.irl_model.step], feed_dict={
                            algo.irl_model.act_t: act_batch,
                            algo.irl_model.obs_t: obs_batch,
                            algo.irl_model.labels: labels,
                            algo.irl_model.lprobs: lprobs_batch,
                            algo.irl_model.lr: lr
                        })
            
                        it.record('loss', loss)
                        if it.heartbeat:
                            print(it.itr_message())
                            mean_loss = it.pop_mean('loss')
                            print('\tLoss:%f' % mean_loss)
                        if logger:
                            energy = tf.get_default_session().run(algo.irl_model.energy,
                                                                        feed_dict={algo.irl_model.act_t: acts, algo.irl_model.obs_t: obs})
                            logger.record_tabular('IRLAverageEnergy', np.mean(energy))
                            logger.record_tabular('IRLAverageLogQtau', np.mean(path_probs))
                            logger.record_tabular('IRLMedianLogQtau', np.median(path_probs))
                
                            energy = tf.get_default_session().run(algo.irl_model.energy,
                                                                        feed_dict={algo.irl_model.act_t: expert_acts, algo.irl_model.obs_t: expert_obs})
                            logger.record_tabular('IRLAverageExpertEnergy', np.mean(energy))
                            #logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy-logZ))
                            logger.record_tabular('IRLAverageExpertLogQtau', np.mean(expert_probs))
                            logger.record_tabular('IRLMedianExpertLogQtau', np.median(expert_probs))

                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
        
                    logger.record_tabular('IRLLoss', mean_loss)
                    algo.__irl_params = algo.irl_model.get_params()

#                    probs = algo.irl_model.eval(paths, gamma=algo.discount, itr=itr)

            #        logger.record_tabular('IRLRewardMean', np.mean(probs))
            #        logger.record_tabular('IRLRewardMax', np.max(probs))
            #        logger.record_tabular('IRLRewardMin', np.min(probs))
            
                    obs, acts = algo.irl_model.extract_paths(paths)

                    energy  = tf.get_default_session().run(algo.irl_model.energy,
                                                                feed_dict={algo.irl_model.act_t: acts, algo.irl_model.obs_t: obs})
                    energy = -energy[:,0] 
                    probs = algo.irl_model.unpack(energy, paths)



                    if algo.irl_model.score_trajectories:
                        # TODO: should I add to reward here or after advantage computation?
                        for i, path in enumerate(paths):
                            path['rewards'][-1] += algo.irl_model_wt * probs[i]
                    else:
                        for i, path in enumerate(paths):
                            path['rewards'] += algo.irl_model_wt * probs[i]
                    
                
    
    
    
    
                
                
                returns.append(algo.log_avg_returns(paths))
                samples_data = algo.process_samples(itr, paths)

                logger.log("Logging diagnostics...")
                algo.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                algo.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = algo.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if algo.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if algo.plot:
                    algo.update_plot()
                    if algo.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        algo.shutdown_worker()

#import pickle
#
#with open("algo.pickle","wb") as f:
#    pickle.dump(algo, f)
#    
#f = open('algo.obj', 'wb')
#pickle.dump(algo, f)
        
