import numpy as np
import taichi as ti
from JFA import jfa_solver_2D
from Proposed import proposed_solver_2D
ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)

# runner for JFA
# width, height, sites = 512, 512, np.array(np.random.rand(100, 2), dtype=np.float32)
# voronoi = jfa_solver_2D(width, height, sites)
# init_step=(200,200)
# voronoi.solve_jfa(init_step)
# # voronoi.display() # uncomment for obtaining image of Voronoi 
# ti.profiler.print_kernel_profiler_info()

# runner for proposed algo
width, height, sites = 512, 512, np.array(np.random.rand(100, 2), dtype=np.float32)
voronoi = proposed_solver_2D(width, height, sites)
voronoi.solve_proposed()
# voronoi.display() # uncomment for obtaining image of Voronoi 
ti.profiler.print_kernel_profiler_info()


    





