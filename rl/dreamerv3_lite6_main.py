"""DreamerV3 entrypoint that adds a custom 'lite6' env suite.

We keep this file inside the repo (tracked) instead of patching site-packages.
Usage example:
  source rl/.venv/bin/activate
  python rl/dreamerv3_lite6_main.py --task lite6_reach --run.steps 2000 --run.envs 1
"""

import importlib
import os
import pathlib
import sys
from functools import partial as bind

# Ensure repo root is on sys.path so we can import lite6_rpc_env.
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


def main(argv=None):
  from dreamerv3.agent import Agent
  [elements.print(line) for line in Agent.banner]

  # Load upstream DreamerV3 configs
  import importlib
  mod = importlib.import_module("dreamerv3.main")
  folder = pathlib.Path(mod.__file__).parent
  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)

  # Inject env.lite6 defaults so CLI flags are accepted.
  configs['defaults'].setdefault('env', {})
  configs['defaults']['env'].setdefault('lite6', {
      'host': '127.0.0.1',
      'port': 5555,
      'port_base': 5555,
      'timeout': 30,
      'logdir': '',
      'video_fps': 30,
      'video_w': 640,
      'video_h': 480,
      'video_seconds': 20,
      'video_every': 0,
      'download_dir': '~/Downloads',
      'download_prefix': 'robotarm training video',
  })
  # Make sure newly-added keys exist even if lite6 dict already existed.
  configs['defaults']['env'].setdefault('lite6', {})
  configs['defaults']['env']['lite6'].setdefault('video_every', 0)
  configs['defaults']['env']['lite6'].setdefault('port_base', 5555)
  configs['defaults']['env']['lite6'].setdefault('download_dir', '~/Downloads')
  configs['defaults']['env']['lite6'].setdefault('download_prefix', 'robotarm training video')

  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)

  # Normalize logger.outputs when passed as a single string like "[jsonl,scope]".
  outs = list(config.logger.outputs)
  if len(outs) == 1 and isinstance(outs[0], str) and outs[0].lstrip().startswith("["):
    s = outs[0].strip()
    if s.startswith("[") and s.endswith("]"):
      inner = s[1:-1].strip()
      outs = [x.strip() for x in inner.split(",") if x.strip()]
      config = config.update(logger={**dict(config.logger), "outputs": outs})

  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  # If env.lite6.logdir not set, default to run logdir so the worker can save episode videos there.
  try:
    l6 = dict(config.env.get('lite6', {}))
    if not str(l6.get('logdir','')):
      l6['logdir'] = config.logdir
      config = config.update(env={**dict(config.env), 'lite6': l6})
  except Exception:
    pass

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)
  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from dreamerv3.agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity
  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)
  return embodied.replay.Replay(**kwargs)


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)
  return stream
def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)

  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dmc': 'embodied.envs.dmc:DMC',
      'lite6': 'lite6_rpc_env:Lite6RPCEnv',
  }[suite]

  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)

  kwargs = config.env.get(suite, {})
  # Ensure worker knows where to save per-episode videos
  if suite == 'lite6' and not str(kwargs.get('logdir','')):
    kwargs = {**kwargs, 'logdir': config.logdir}
  # lite6: per-env port mapping and only env0 exports videos to Downloads
  if suite == 'lite6':
    pb = kwargs.get('port_base', None)
    if pb is not None:
      try:
        kwargs = {**kwargs, 'port': int(pb) + int(index)}
      except Exception:
        pass
    if int(index) != 0:
      # Disable video/export on nonzero envs to avoid spamming.
      try:
        kwargs = {**kwargs, 'video_every': 0, 'download_dir': ''}
      except Exception:
        pass

  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'

  env = ctor(task, **kwargs)

  # Wrap like upstream
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()