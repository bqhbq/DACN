from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SCRIPT = REPO_ROOT / "tools" / "generator" / "generate_grid_scenario.py"

def generate_roadnet(
  rows: int,
  cols: int,
  output_dir: str | Path,
  roadnet_name: str = "roadnet.json",
  num_left_lanes: int | None = None,
  num_straight_lanes: int | None = None,
  num_right_lanes: int | None = None,
  row_distance: int | None = None,
  column_distance: int | None = None,
  tl_plan: bool = True,
  turn: bool = False,
) -> Path:
  """Sinh roadnet cho lưới rows x cols.

  Args:
    turn: Giữ tương thích với lời gọi cũ; tùy chọn này chỉ áp dụng khi sinh flow.
  """
  if not GENERATOR_SCRIPT.is_file():
      raise FileNotFoundError(GENERATOR_SCRIPT)

  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)  # tương đương mkdir -p

  command = [
      sys.executable, str(GENERATOR_SCRIPT),
      str(rows), str(cols),
      "--dir", str(output_dir),
      "--roadnetFile", roadnet_name,
      "--flowFile", "temp_flow.json"
  ]

  optional = {
      "--numLeftLanes": num_left_lanes,
      "--numStraightLanes": num_straight_lanes,
      "--numRightLanes": num_right_lanes,
      "--rowDistance": row_distance,
      "--columnDistance": column_distance,
  }
  for flag, value in optional.items():
      if value is not None:
          command += [flag, str(value)]

  if tl_plan:
      command.append("--tlPlan")
  if turn:
      command.append("--turn")          

  result = subprocess.run(
      command, capture_output=True, text=True, check=False,
      cwd=GENERATOR_SCRIPT.parent,
  )
  if result.stdout:
      print(result.stdout)
  if result.returncode != 0:
      raise RuntimeError(f"generate_grid_scenario.py failed:\n{result.stderr}")

  roadnet_path = output_dir / roadnet_name
  if not roadnet_path.is_file():
      raise RuntimeError(
          "generate_grid_scenario.py completed without creating "
          f"the roadnet file: {roadnet_path}"
      )
  print(f"Roadnet: {roadnet_path}")
  return roadnet_path

