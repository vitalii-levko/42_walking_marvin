#!/usr/local/bin/python3
import argparse
import gym
import numpy as np
import pickle
from gym.wrappers import Monitor
import logging
import multiprocessing as mp
from datetime import datetime as dt


def wrap_env(env):
  env = Monitor(env, './video', force=True)
  
  return env


def _parse_args():
  pool = mp.cpu_count()
  seed = 1
  epochs = 100
  print_step = 1
  sigma = 0.1
  alpha = 0.03
  decay = 0.999
  population_size = 42
  max_play_steps = 2424
  max_train_steps = 2424
  layer = [24, 16]
  parser = argparse.ArgumentParser(
    description='A depressed robot training to walk')
  parser.add_argument(
    '-pool',
    metavar='NUMBER',
    type=int,
    default=pool,
    choices=range(1, 10 * pool + 1),
    help='set multiprocessing number for Mravin weights train')
  parser.add_argument(
    '-seed',
    metavar='NUMBER',
    type=int,
    default=seed,
    help='set seed number for random weights generator')
  parser.add_argument(
    '-print',
    metavar='NUMBER',
    type=int,
    default=print_step,
    choices=range(1, 10001),
    help='set print step number')
  parser.add_argument(
    '-epochs',
    metavar='NUMBER',
    type=int,
    default=epochs,
    choices=range(1, 10001),
    help='set number of epochs for Marvin weights train')
  parser.add_argument(
    '-sigma',
    metavar='NUMBER',
    type=float,
    default=sigma,
    help='set sigma number for population weights generator')
  parser.add_argument(
    '-alpha',
    metavar='NUMBER',
    type=float,
    default=alpha,
    help='set alpha number for learning rate')
  parser.add_argument(
    '-decay',
    metavar='NUMBER',
    type=float,
    default=decay,
    help='set decay number for next alfa')
  parser.add_argument(
    '-pop',
    metavar='NUMBER',
    type=int,
    default=population_size,
    choices=range(1, 101),
    help='set population number for Marvin weights train')
  parser.add_argument(
    '-play_steps',
    metavar='NUMBER',
    type=int,
    default=max_play_steps,
    choices=range(1, 10001),
    help='set max number of play steps')
  parser.add_argument(
    '-train_steps',
    metavar='NUMBER',
    type=int,
    default=max_train_steps,
    choices=range(1, 10001),
    help='set max number of train steps')
  parser.add_argument(
    '-layer',
    metavar='NUMBER',
    type=int,
    default=layer,
    choices=range(1, 101),
    nargs='+',
    help='set hidden layer size')
  parser.add_argument(
    '-log',
    metavar='FILE',
    type=argparse.FileType('w'),
    help='write log to a file')
  parser.add_argument(
    '-l',
    '--load',
    metavar='FILE',
    type=argparse.FileType('rb'),
    help='load weights for Marvin agent from a file; skip training process if this option is specified')
  parser.add_argument(
    '-s',
    '--save',
    metavar='FILE',
    type=argparse.FileType('wb'),
    help='save weights to a file after running the program')
  parser.add_argument(
    '-t',
    '--train',
    action='store_true',
    help='run only training process; skip walking process; if load option specified, train with loaded weights')
  parser.add_argument(
    '-w',
    '--walk',
    action='store_true',
    help='display only walking process')
  parser.add_argument(
    '-norm',
    action='store_true',
    help='normalize output value for prediction')
  parser.add_argument(
    '-detail',
    action='store_true',
    help='print detailed log')
  parser.add_argument(
    '-silent',
    action='store_true',
    help='do not print log for each train epoch and play step')
  parser.add_argument(
    '-render',
    action='store_true',
    help='render video of marvin play')
  parser.add_argument(
    '-capture',
    action='store_true',
    help='capture rendering to video file')
  parser.add_argument(
    '-current',
    action='store_true',
    help='print current or default arguments values')

  return parser.parse_args()


def _config_logging(args):
  if args.log is None:
    logging.basicConfig(filename=None, level=logging.INFO, format='%(asctime)s %(message)s')
  else:
    logging.basicConfig(stream=args.log, level=logging.INFO, format='%(asctime)s %(message)s')


def _set_layers_size(args):
  layers_size = [24, 4]
  layers_size[1:-1] = args.layer

  return layers_size


def _generate_random_weights(args):
  weights_layers_size = _set_layers_size(args)
  seed = args.seed
  weights = []
  np.random.seed(seed)

  for i in range(len(weights_layers_size) - 1):
    weights_layer = np.random.randn(weights_layers_size[i], weights_layers_size[i + 1])
    weights.append(weights_layer)

  return np.array(weights), weights_layers_size


def _generate_population(population_size, weights_layers_size):
  population = []

  for _ in range(population_size):
    normal_distribution = []

    for i in range(len(weights_layers_size) - 1):
      weights_layer = np.random.randn(weights_layers_size[i], weights_layers_size[i + 1])
      normal_distribution.append(weights_layer)

    population.append(normal_distribution)

  return np.array(population)


def _predict_action(observation, weights, norm):
  out = np.expand_dims(observation.flatten(), 0)
  
  if norm:
    out = out / np.linalg.norm(out)

  for weights_layer in weights:
    out = np.dot(out, weights_layer)
    out = np.tanh(out)

  return out[0]


def _get_reward(env, weights, max_train_steps, norm):
  env_local = None

  if env is None:
    env = gym.make('Marvin-v0')
    env_local = True

  total_reward = timesteps = 0
  observation = env.reset()
  done = False

  while not done and timesteps < max_train_steps:
    action = _predict_action(observation, weights, norm)
    observation, reward, done, _ = env.step(action)
    total_reward += reward
    timesteps += 1

  if env_local is None:
    env.close()

  return total_reward, timesteps


def _get_population_weights(pop, weights, sigma):
  population_weights = []

  for index, pop_layer in enumerate(pop):
    weights_layer = weights[index] + sigma * pop_layer
    population_weights.append(weights_layer)

  return np.array(population_weights)


def _worker_process(arg):
  env = None
  fp, population_weights, max_train_steps, norm = arg
  ret, _ = fp(env, population_weights, max_train_steps, norm)

  return ret


def _get_rewards(env, population_size, population, weights, sigma, pool, max_train_steps, norm, detail, silent, walk_only):
  if pool:

    population_weights = []
    for i in range(population_size):
      population_weights.append(_get_population_weights(population[i], weights, sigma))
    worker_args = ((_get_reward, pop, max_train_steps, norm) for pop in population_weights)
    rewards = np.array(pool.map(_worker_process, worker_args))
    
  else:
    rewards = np.zeros(population_size)

    for i in range(population_size):
      population_weights = _get_population_weights(population[i], weights, sigma)
      rewards[i], timesteps = _get_reward(env, population_weights, max_train_steps, norm)
    
      if not walk_only and not silent and detail:
        logging.info(f'>>> train population {i + 1} with reward {rewards[i]} and {timesteps} of {max_train_steps} max train steps')

  return rewards


def _update_weights(population_size, population, weights, rewards, sigma, alpha):
  updated_weights = []
  rewards_mean = np.mean(rewards)
  rewards_std = np.std(rewards)

  if rewards_std == 0:
    return weights

  rewards_norm = (rewards - rewards_mean) / rewards_std

  for index, weights_layer in enumerate(weights):
    population_layer = np.array([pop[index] for pop in population])
    factor = alpha / (population_size * sigma)
    updated_layer = weights_layer + factor * np.dot(population_layer.T, rewards_norm).T
    updated_weights.append(updated_layer)

  return np.array(updated_weights)


def _train_marvin_weights(args):
  weights, weights_layers_size = _generate_random_weights(args)

  if args.load is not None:
    weights = pickle.load(args.load)

  env = gym.make('Marvin-v0')
  sigma = args.sigma
  alpha = args.alpha
  decay = args.decay
  population_size = args.pop
  processes = args.pool
  epochs = args.epochs
  max_train_steps = args.train_steps
  walk_only = args.walk
  print_step = args.print
  detail =  args.detail
  silent =  args.silent
  norm = args.norm
  observation = env.reset()

  if processes > 1:
    pool = mp.Pool(processes=processes)
  else:
    pool = False

  if not walk_only and detail:
    logging.info(f'>>> start Marvin weights train with following arguments')
    _list_args(args)

  for epoch in range(epochs):
    t1 = dt.now()
    population = _generate_population(population_size, weights_layers_size)
    rewards = _get_rewards(env, population_size, population, weights, sigma, pool, max_train_steps, norm, detail, silent, walk_only)
    weights = _update_weights(population_size, population, weights, rewards, sigma, alpha)
    alpha *= decay
    reward, timesteps = _get_reward(env, weights, max_train_steps, norm)

    if not walk_only and (not silent or detail) and (epoch + 1) % print_step == 0:
      logging.info(f'>>> train epoch {epoch + 1} - time {dt.now() - t1} - reward {reward} - timesteps {timesteps} of {max_train_steps}')

  if not walk_only and detail:
    logging.info(f'>>> end Marvin weights train after {epoch + 1} epoch(-s) with final reward {reward}')
    logging.info(f'>>> Marvin parameters alpha {alpha}, weights layers {weights_layers_size}, multiprocessing {processes if pool else 1}')

  if args.save is not None:
    pickle.dump(weights, args.save)

  if pool:
    pool.close()
    pool.join()

  env.close()

  return weights


def _play_marvin(weights, args):
  max_play_steps = args.play_steps
  render = args.render
  detail = args.detail
  silent = args.silent
  walk_only = args.walk
  norm = args.norm
  if args.capture:
    env = wrap_env(gym.make('Marvin-v0'))
  else:
    env = gym.make('Marvin-v0')
  play_reward = play_steps = 0
  done = False
  observation = env.reset()

  if not silent or detail or walk_only:
   logging.info(f'>>> start Marvin play with {max_play_steps} max steps')

  while not done and play_steps < max_play_steps:

    if render:
      env.render()

    action = _predict_action(observation, weights, norm)
    observation, reward, done, _ = env.step(action)
    play_reward += reward
    play_steps += 1
    if not silent and detail:
      logging.info(f'>>> play step {play_steps} - reward {reward}')

  if not silent or detail or walk_only:
    logging.info(f'>>> end Marvin play after {play_steps} steps with total reward {play_reward} and done status {done}')
  
  env.close()


def _list_args(args):
  logging.info(f'>>> ' + ' | '.join([f'{k}: {v.name}' if (k == 'log' or k == 'load' or k == 'save') and v else f'{k}: {v}' for k, v in vars(args).items()]))


def _main():
  args = _parse_args()
  _config_logging(args)
  
  if args.current:
    logging.info(f'>>> current or default arguments values are')
    _list_args(args)

  if args.train or args.load is None:

    try:
      t1 = dt.now()
      weights = _train_marvin_weights(args)      
      if not args.walk and args.detail:
        logging.info(f'>>> train time {dt.now() - t1}')
    except Exception as err:
      logging.info(f'>>> error in Marvin train with following arguments - {err}:')
      _list_args(args)
      return

  elif args.load is not None:

    try:
      weights = pickle.load(args.load)
    except Exception as err:
      logging.info(f'>>> error loading file `{args.load.name}` - {err}')
      return

  if not args.train:
    try:
      t1 = dt.now()
      _play_marvin(weights, args)
      if args.detail:
        logging.info(f'>>> play time {dt.now() - t1}')
    except Exception as err:
      logging.info(f'>>> error in Marvin play with following arguments - {err}:')
      _list_args(args)
      return


if __name__ == '__main__':
  _main()
