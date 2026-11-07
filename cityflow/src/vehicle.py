from dataclasses import dataclass


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

  def to_cityflow_dict(self) -> dict:
    return {
      "length": self.length,
      "width": self.width,
      "maxPosAcc": self.max_pos_acc,
      "maxNegAcc": self.max_neg_acc,
      "usualPosAcc": self.usual_pos_acc,
      "usualNegAcc": self.usual_neg_acc,
      "minGap": self.min_gap,
      "maxSpeed": self.max_speed,
      "headwayTime": 1.5,
    }


class VehicleFleet:
  BICYCLE = VehicleType(
    name="bicycle",
    length=1.8,
    width=0.6,
    max_pos_acc=1.5,
    max_neg_acc=4.0,
    usual_pos_acc=1.0,
    usual_neg_acc=2.5,
    min_gap=1.0,
    max_speed=4.2,
  )

  MOTORCYCLE = VehicleType(
    name="motorcycle",
    length=2.0,
    width=0.8,
    max_pos_acc=3.5,
    max_neg_acc=5.5,
    usual_pos_acc=2.5,
    usual_neg_acc=3.5,
    min_gap=1.5,
    max_speed=11.1,
  )

  CAR = VehicleType(
    name="car",
    length=4.5,
    width=1.8,
    max_pos_acc=3.0,
    max_neg_acc=5.0,
    usual_pos_acc=2.0,
    usual_neg_acc=3.0,
    min_gap=2.5,
    max_speed=13.9,
  )

  BUS = VehicleType(
    name="bus",
    length=12.0,
    width=2.5,
    max_pos_acc=1.2,
    max_neg_acc=3.5,
    usual_pos_acc=0.8,
    usual_neg_acc=2.0,
    min_gap=3.5,
    max_speed=11.1,
  )

  TRUCK = VehicleType(
    name="truck",
    length=8.5,
    width=2.4,
    max_pos_acc=1.5,
    max_neg_acc=3.8,
    usual_pos_acc=1.0,
    usual_neg_acc=2.2,
    min_gap=3.0,
    max_speed=10.0,
  )
