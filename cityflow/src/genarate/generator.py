from __future__ import annotations

from pathlib import Path

from .config_generator import ConfigGenerator
from .flow_generator import FlowGenerator
from .network_converter import NetworkConverter

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
      in_file=self.in_file,
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
        roadnet_json_file=self.roadnet_json_file,
        flow_json_file=self.flow_json_file,
        config_json_file=self.cityflow_config_file,
    )

  def generate(self):
    # 1. Chuyển đổi mạng lưới
    self.converter.osm_to_net_xml()
    self.converter.net_xml_to_cityflow()
    
    # 2. Sinh flow
    self.flow_gen.generate()
    
    # 3. Sinh config
    self.config_gen.generate()
