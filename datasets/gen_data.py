import os
import time
import sys
import numpy as np
from ase import Atoms, Atom, units
from ase.build import molecule
from ase.visualize import view
import ase.io
from ase.calculators.emt import EMT

# from asap3 import EMT
from ase.build import fcc110, fcc111, fcc100, add_adsorbate, molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet, Langevin, nvtberendsen
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms, Hookean
# from asap3 import EMT

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork


def generate_data(count, filename, temp, hook, cons_t=False):
    """Generates test or training data with a simple MD simulation."""
    traj = ase.io.Trajectory(filename, "w")
    slab = fcc100("Cu", size=(3, 3, 3))
    ads = molecule("CO")
    add_adsorbate(slab, ads, 5, offset=(1, 1))
    cons = FixAtoms(
        indices=[atom.index for atom in slab if (atom.tag == 2 or atom.tag == 3)]
    )
    if hook:
        cons2 = Hookean(a1=28, a2=27, rt=1.58, k=10.0)
        slab.set_constraint([cons, cons2])
    else:
        slab.set_constraint(cons)
    slab.center(vacuum=13., axis=2)
    slab.set_calculator(EMT())
    # slab.set_pbc(True)
    # dyn = QuasiNewton(slab, trajectory=(filename[:-5] + "_relax.traj"))
    # dyn.run(fmax=0.05)
    # traj.write(slab)
    if cons_t is True:
        dyn = Langevin(slab, 5 * units.fs, temp * units.kB, 0.002)
        # dyn = nvtberendsen.NVTBerendsen(slab, 1 * units.fs, temp * units.kB,
                # taut=0.5*1000*units.fs)
    else:
        dyn = VelocityVerlet(slab, dt=1.0 * units.fs)
    for step in range(count - 1):
        dyn.run(20)
        traj.write(slab)


generate_data(1000, "COCu/COCu_pbc_300.traj", temp=300.0, hook=False, cons_t=True)
# generate_data(300, "COCu/COCu_pbc.traj", temp=300.0, hook=False, cons_t=False)
# generate_data(300, "COCu/COCu_pbc_conT.traj", temp=300.0, hook=False, cons_t=True)
# generate_data(300, "COCu/COCu_pbc_hook.traj", temp=300.0, hook=True, cons_t=False)
# generate_data(300, "COCu/COCu_pbc_hook_conT.traj", temp=300.0, hook=True, cons_t=True)

