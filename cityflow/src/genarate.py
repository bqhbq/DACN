from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict

from src.genarate import NetworkConverter, ConfigGenerator, FlowGenerator
from .vehicle import VehicleFleet, VehicleType


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONVERTER_PATH = REPO_ROOT / "externals" / "CityFlow" / "tools" / "converter" 

class Generator:
  def __init__(self, 
               osm_file: str, 
               converter_version: str = "converter",
  ):
    self.in_file = Path(osm_file).resolve()
    self.out_dir = (Path("build/generator_cityflow") / self.in_file.stem).resolve()
    self.out_dir.mkdir(parents=True, exist_ok=True)
    
    self.cityflow_config_file = self.out_dir / "config.json"
    self.roadnet_json_file = self.out_dir / "roadnet.json"
    self.flow_json_file = self.out_dir / "flow.json"
    self.net_xml_file = self.out_dir / "temp.net.xml"

    self.converter = NetworkConverter(
      osm_file=self.in_file,
      net_xml_file=self.net_xml_file,
      roadnet_json_file=self.roadnet_json_file,
      converter_version=converter_version,
      out_dir=self.out_dir,
    )
    self.flow_gen = FlowGenerator(
        roadnet_json_file=self.roadnet_json_file,
        flow_json_file=self.flow_json_file,
    )
    self.config_gen = ConfigGenerator(
        config_file=self.cityflow_config_file,
        out_dir=self.out_dir,
        roadnet_filename=self.roadnet_json_file.name,
        flow_filename=self.flow_json_file.name,
    )

  def generate(self):
    self._osm_to_net_xml()
    self._net_xml_to_cityflow()
    self._generate_flow()
    self._generate_config()
  
