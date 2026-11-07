from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import sumo

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONVERTER_PATH = REPO_ROOT / "externals/CityFlow/tools/converter/" 


class NetworkConverter:
  def __init__(self, 
               in_file, 
               net_xml_file,
               roadnet_json_file,
               converter_version="converter",
               out_dir=None
  ):
    self.in_file = Path(in_file).resolve()
    self.out_dir = Path(out_dir).resolve()
    self.converter_version = converter_version
    
    self.net_xml_file = net_xml_file
    self.roadnet_json_file = roadnet_json_file
    
    self.converter_version = converter_version
    if self.converter_version == "converter_v2":
      self.converter_path = DEFAULT_CONVERTER_PATH / "converter_v2.py"
    else:
      self.converter_path = DEFAULT_CONVERTER_PATH / "converter.py"

  @staticmethod
  def _run_command(command, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)
    print(f"Command: {' '.join(command)}")
    if result.stdout:
      print(f"Output: {result.stdout}")
    if result.stderr:
      print(f"Error: {result.stderr}")

  def _get_netconvert_bin(self):
    # 1. Thử lấy từ gói sumo của python
    try:
      sumo_home = Path(sumo.SUMO_HOME)
      candidate = sumo_home / "bin" / "netconvert"
      if candidate.is_file():
        return str(candidate)
    except (AttributeError, TypeError):
      pass
    
    # 2. Thử tìm trên PATH hệ thống
    netconvert_path = shutil.which("netconvert")
    if netconvert_path:
      return netconvert_path
    
    raise FileNotFoundError("Not found netconvert binary. Please ensure SUMO is installed and netconvert is in your PATH.")

  def _osm_to_net_xml(self):
    netconvert_bin = self._get_netconvert_bin()
    command = [
      netconvert_bin,
      "--osm-files", str(self.in_file),
      "--proj.utm",
      "--tls.discard-loaded",
      "--output-file", str(self.net_xml_file),
    ]

    self._run_command(command=command, cwd=self.out_dir)
    print(f"Đã chuyển đổi OSM sang SUMO net XML: {self.net_xml_file}")

  def _run_converter_v1(self):
    command = [
      sys.executable,
      str(self.converter_path),
      "--sumonet", str(self.net_xml_file),
      "--cityflownet", str(self.roadnet_json_file),
    ]
    self._run_command(command)

  def _run_converter_v2(self):
    with tempfile.TemporaryDirectory(prefix="cityflow_converter_") as temp_dir:
      temp_root = Path(temp_dir)
      data_dir = temp_root / "data" / "raw_data"
      runner_dir = temp_root / "runner"
      data_dir.mkdir(parents=True)
      runner_dir.mkdir()
      shutil.copy2(self.net_xml_file, data_dir / "network.net.xml")

      runner = runner_dir / "run_converter.py"
      runner.write_text(
        "import importlib.util\n"
        "from types import SimpleNamespace\n"
        f"spec = importlib.util.spec_from_file_location('converter_v2', {str(self.converter_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "args = SimpleNamespace(\n"
        "  or_sumonet='network.net.xml',\n"
        "  cityflownet='roadnet.json'\n"
        ")\n"
        "module.sumo2cityflow_net(args)\n",
        encoding="utf-8",
      )
      self._run_command([sys.executable, str(runner)], cwd=runner_dir)
      shutil.copy2(data_dir / "roadnet.json", self.roadnet_json_file)

  def _net_xml_to_cityflow(self):
    if self.converter_version == "converter_v2":
      self._run_converter_v2()
      print(f"Done converting SUMO net XML to CityFlow_v2 roadnet JSON: {self.roadnet_json_file}")
    else:
      self._run_converter_v1()
      print(f"Done converting SUMO net XML to CityFlow_v1 roadnet JSON: {self.roadnet_json_file}")

  def convert(self):
    self._osm_to_net_xml()
    self._net_xml_to_cityflow()