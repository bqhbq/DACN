from src.simulate import Simulation
from src.agent import TrafficLightAgent

if __name__ == "__main__":
  simulation = Simulation(
    "build/generator_cityflow/sample_map_2/config.json",
    total_steps=2000
  )
  
  agents = [
    # TrafficLightAgent(simulation.engine, "intersection_0_0", num_phases=2),
  ]

  try:
    simulation.run(agents)
  finally:
    simulation.close()
  
