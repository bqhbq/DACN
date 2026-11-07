from pathlib import Path
import cityflow

class Simulation:
  def __init__(
    self, 
    config_path: str | Path , 
    total_steps: int
  ):
    self.config_path = Path(config_path).resolve()
    self.total_steps = total_steps
    self.render_dir = self.config_path.parent / "render"
    self.render_dir.mkdir(parents=True, exist_ok=True)
    for replay_file in ("replay.txt", "replay_roadnet.json"):
      replay_path = self.render_dir / replay_file
      if replay_path.exists():
        replay_path.unlink()
    self.engine = cityflow.Engine(str(self.config_path), thread_num=1)
    
  @staticmethod
  def visualize():
    import errno
    import functools
    import http.server
    import socketserver
    import webbrowser

    PORT = 8080
    DIRECTORY = str(Path("externals") / "CityFlow" / "frontend")
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIRECTORY)

    class ReusableTCPServer(socketserver.TCPServer):
      allow_reuse_address = True

    try:
      with ReusableTCPServer(("", PORT), Handler) as httpd:
        print(f"CityFlow Visualizer đang chạy tại http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        try:
          httpd.serve_forever()
        except KeyboardInterrupt:
          print("\nĐã tắt server.")
    except OSError as error:
      if error.errno != errno.EADDRINUSE:
        raise
      raise RuntimeError(
        f"Cổng {PORT} đang được sử dụng. Hãy đóng visualizer cũ hoặc đổi PORT."
      ) from error
      
  def run(self, agents):
    phase_duration = 10
    for step in range(1, self.total_steps + 1):
      
      # if (step - 1) % phase_duration == 0:
      #   for agent in agents:
      #     state = agent.observe()
      #     phase_id = agent.act(state)
      #     self.engine.set_tl_phase(agent.intersection_id, phase_id)
      
      # Tiến hành bước mô phỏng tiếp theo
      self.engine.next_step()

      if step % 100 == 0:
        print(f"Step: {step}")
        print(f"Time: {self.engine.get_current_time()}")
        print(f"Vehicles: {self.engine.get_vehicle_count()}")
        # print(f"Avg travel time: {self.engine.get_average_travel_time()}")
        # print(f"Vehicle IDs: {self.engine.get_vehicles(include_waiting=True)}")
        # print(f"Lane vehicle count: {self.engine.get_lane_vehicle_count()}")
        # print(f"Lane waiting count: {self.engine.get_lane_waiting_vehicle_count()}")
        # print(f"Lane vehicles: {self.engine.get_lane_vehicles()}")
        
  def close(self):
    """Close CityFlow so replay output is flushed before the process exits."""
    del self.engine
    self.visualize()

