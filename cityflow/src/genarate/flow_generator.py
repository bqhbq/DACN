from __future__ import annotations

import json
import random
from pathlib import Path
from collections.abc import Iterable
from typing import Any

if __package__:
  from ..vehicle import VehicleFleet, VehicleType
else:
  from src.vehicle import VehicleFleet, VehicleType

class FlowGenerator:
  def __init__(
    self,
    roadnet_json_file: str | Path,
    flow_json_file: str | Path,
  ):
    self.roadnet_json_file = Path(roadnet_json_file).resolve()
    self.flow_json_file = Path(flow_json_file).resolve()

  def _build_topology(self) -> dict[str, list[str]]:
    with self.roadnet_json_file.open(encoding="utf-8") as roadnet_file:
      roadnet: dict[str, Any] = json.load(roadnet_file)

    topology: dict[str, set[str]] = {}
    for intersection in roadnet.get("intersections", []):
      for road_link in intersection.get("roadLinks", []):
        start_road = road_link["startRoad"]
        end_road = road_link["endRoad"]
        topology.setdefault(start_road, set()).add(end_road)

    return {
      road: sorted(next_roads)
      for road, next_roads in topology.items()
    }

  def _find_routes(
    self,
    topology: dict[str, list[str]],
    roadnet: dict[str, Any],
  ) -> list[list[str]]:
    intersections = {
      intersection["id"]: intersection
      for intersection in roadnet.get("intersections", [])
    }
    entry_roads: list[str] = []
    exit_roads: set[str] = set()
    for road in roadnet.get("roads", []):
      start = intersections[road["startIntersection"]]
      end = intersections[road["endIntersection"]]
      if start.get("virtual", False) and not end.get("virtual", False):
        entry_roads.append(road["id"])
      if not start.get("virtual", False) and end.get("virtual", False):
        exit_roads.add(road["id"])

    routes: list[list[str]] = []

    def visit(route: list[str]) -> None:
      current_road = route[-1]
      if current_road in exit_roads:
        routes.append(route.copy())
        return
      for next_road in topology.get(current_road, []):
        if next_road not in route:
          visit(route + [next_road])

    for entry_road in sorted(entry_roads):
      visit([entry_road])
    return routes

  @staticmethod
  def _vehicle_types(vehicle_fleet: VehicleFleet | VehicleType | Iterable[VehicleType]) -> list[VehicleType]:
    if isinstance(vehicle_fleet, VehicleType):
      return [vehicle_fleet]
    if isinstance(vehicle_fleet, type) and issubclass(vehicle_fleet, VehicleFleet):
      return [
        value for value in vars(vehicle_fleet).values()
        if isinstance(value, VehicleType)
      ]
    return list(vehicle_fleet)

  def generate(
    self,
    vehicle_fleet: VehicleFleet | VehicleType | Iterable[VehicleType] | None = None,
  ) -> list[dict[str, Any]]:
    with self.roadnet_json_file.open(encoding="utf-8") as roadnet_file:
      roadnet: dict[str, Any] = json.load(roadnet_file)

    topology = self._build_topology()
    routes = self._find_routes(topology, roadnet)
    if not routes:
      raise ValueError("No route connects a virtual entry to a virtual exit.")

    fleet = VehicleFleet.CAR if vehicle_fleet is None else vehicle_fleet
    vehicle_types = self._vehicle_types(fleet)
    if not vehicle_types:
      raise ValueError("vehicle_fleet must contain at least one VehicleType.")

    flow: list[dict[str, Any]] = []
    for vehicle_type in vehicle_types:
      for route in routes:
        selected_vehicle = random.choice(vehicle_types)
        flow.append({
          "vehicle": selected_vehicle.to_cityflow_dict(),
          "route": route,
          "interval": round(random.uniform(10.0, 15.0), 2),
          "startTime": 0,
          "endTime": -1,
        })

    self.flow_json_file.parent.mkdir(parents=True, exist_ok=True)
    with self.flow_json_file.open("w", encoding="utf-8") as flow_file:
      json.dump(flow, flow_file, indent=2)
      flow_file.write("\n")
    return flow

if __name__ == "__main__":
  roadnet = "build/generator_cityflow/sample_map_3/roadnet.json"
  flow = "build/generator_cityflow/sample_map_3/flow.json"

  generator = FlowGenerator(
    roadnet_json_file=roadnet,
    flow_json_file=flow,
  )
  generator.generate()