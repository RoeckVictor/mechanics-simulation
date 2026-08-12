import sys
import os

# keeps imports working regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from ui.app import App
from sims.projectile import ProjectileSimulation
from sims.pendulum import PendulumSimulation
from sims.incline import InclineSimulation
from sims.spring_mass import SpringMassSimulation
from sims.double_pendulum import DoublePendulumSimulation
from sims.orbit import OrbitSimulation

SIMULATIONS = [
    ProjectileSimulation(),
    PendulumSimulation(),
    InclineSimulation(),
    SpringMassSimulation(),
    DoublePendulumSimulation(),
    OrbitSimulation(),
]


def main():
    app = App(SIMULATIONS)
    app.run()


if __name__ == "__main__":
    main()
