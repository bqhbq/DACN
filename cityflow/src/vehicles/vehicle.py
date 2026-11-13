from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VehicleType:
  name: str
  length: float
  width: float
  max_pos_acc: float
  max_neg_acc: float
  usual_pos_acc: float
  usual_neg_acc: float
  min_gap: float
  max_speed: float
  headway_time: float = 1.5

  def to_dict(self) -> dict[str, Any]:
    """Chuyển đổi sang schema camelCase theo đúng yêu cầu của CityFlow C++ parser."""
    return {
      "length": self.length,
      "width": self.width,
      "maxPosAcc": self.max_pos_acc,
      "maxNegAcc": self.max_neg_acc,
      "usualPosAcc": self.usual_pos_acc,
      "usualNegAcc": self.usual_neg_acc,
      "minGap": self.min_gap,
      "maxSpeed": self.max_speed,
      "headwayTime": self.headway_time,
    }


class VehicleFleet:
  MOTORCYCLE = VehicleType(
    name="motorcycle",
    length=1.5,
    width=0.5,
    max_pos_acc=3.0,
    max_neg_acc=4.0,
    usual_pos_acc=2.0,
    usual_neg_acc=3.0,
    min_gap=1.0,
    max_speed=20.0,
    headway_time=1.0,
  )

  CAR = VehicleType(
    name="car",
    length=4.5,
    width=1.8,
    max_pos_acc=2.5,
    max_neg_acc=3.5,
    usual_pos_acc=1.5,
    usual_neg_acc=2.5,
    min_gap=2.0,
    max_speed=20.0,
    headway_time=1.5,
  )

  TRUCK = VehicleType(
    name="truck",
    length=8.0,
    width=2.5,
    max_pos_acc=1.5,
    max_neg_acc=4.0,
    usual_pos_acc=1.0,
    usual_neg_acc=3.0,
    min_gap=3.0,
    max_speed=20.0,
    headway_time=2.0,
  )