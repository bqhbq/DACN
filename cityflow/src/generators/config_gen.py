from __future__ import annotations

import json
from pathlib import Path

DEFAULT_INTERVAL = 1.0
DEFAULT_SEED = 42
REPLAY_SUBDIR = "render"


def gen_config(
  dir: str | Path,
  config_name: str,
  roadnet_name: str,
  flow_name: str,
  interval: float = DEFAULT_INTERVAL,
  seed: int = DEFAULT_SEED,
  save_replay: bool = True,
  rl_traffic_light: bool = False,
  lane_change: bool = True,
) -> dict:
  """Tạo config cho CityFlow.

  `dir` là thư mục lưu config, roadnet, flow.
  Replay được lưu trong `dir/render/`.
  Chỉ truyền tên file; CityFlow ghép "dir" + tên file khi load.
  """
  if interval <= 0:
    raise ValueError("interval must be positive")

  base_dir = Path(dir).expanduser().resolve()
  for name in (roadnet_name, flow_name):
    if not (base_dir / name).is_file():
      raise FileNotFoundError(f"File not found: {base_dir / name}")

  if save_replay:
    (base_dir / REPLAY_SUBDIR).mkdir(parents=True, exist_ok=True)

  config = {
    "interval": interval,
    "seed": seed,
    "dir": base_dir.as_posix() + "/",
    "roadnetFile": roadnet_name,
    "flowFile": flow_name,
    "rlTrafficLight": rl_traffic_light,
    "laneChange": lane_change,
    "saveReplay": save_replay,
    "roadnetLogFile": f"{REPLAY_SUBDIR}/replay_roadnet.json",
    "replayLogFile": f"{REPLAY_SUBDIR}/replay.txt",
  }

  with (base_dir / config_name).open("w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)

  return config