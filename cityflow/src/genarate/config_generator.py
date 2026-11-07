from __future__ import annotations
import json
from pathlib import Path

class ConfigGenerator:
  def __init__(self,
               roadnet_json_file: str | Path,
               flow_json_file: str | Path,
               config_json_file: str | Path,
  ):
    self.roadnet_json_file = Path(roadnet_json_file).resolve()
    self.flow_json_file = Path(flow_json_file).resolve()
    self.config_json_file = Path(config_json_file).resolve()
    
  def generate(self,
               interval: int = 1,
               seed: int = 44,
               lane_change: bool = False,
               rl_traffic_light: bool = False,
               save_replay: bool = False,
               thread_num: int = 1,
  ):
    config = {
      "interval": interval,
      "seed": seed,
      "rlTrafficLight": rl_traffic_light,
      "laneChange": lane_change,
      "threadNum": thread_num,
      "saveReplay": save_replay,
      
      "dir": str(self.config_json_file.parent) + "/",
      "roadnetFile": self.roadnet_json_file.name,
      "flowFile": self.flow_json_file.name,
      "roadnetLogFile": "render/replay_roadnet.json",
      "replayLogFile": "render/replay.txt",
    }
    
    with open(self.config_json_file, "w") as f:
      json.dump(config, f, indent=2)
    
    print(f"Generated CityFlow config file: {self.config_json_file}")