


class TrafficLightAgent:
  def __init__(self, engine, intersection_id, num_phases):
    self.engine = engine
    self.intersection_id = intersection_id
    self.num_phases = num_phases

  def observe(self):
    # Lấy trạng thái giao thông hiện tại
    return self.engine.get_lane_waiting_vehicle_count()
  
  def act(self, state):
    self.phase_id = (self.phase_id + 1) % self.num_phases
    return self.phase_id