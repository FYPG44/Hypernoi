import numpy as np
import taichi as ti
from JFA import jfa_solver_2D
from Proposed import proposed_solver_2D
from sqrt_Proposed import proposed_solver_2D_with_sqrt_decomp
ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)
from matplotlib import pyplot as plt

# runner for JFA
# width, height, sites = 512, 512, np.array(np.random.rand(100, 2), dtype=np.float32)
# voronoi = jfa_solver_2D(width, height, sites)
# init_step=(200,200)
# voronoi.solve_jfa(init_step)
# # voronoi.display() # uncomment for obtaining image of Voronoi 
# ti.profiler.print_kernel_profiler_info()

# runner for proposed algo
width, height, seeds, seeds_info = 512, 512, np.array(np.random.rand(128, 2), dtype=np.float32), np.array(np.random.rand(128, 3), dtype=np.float32)
screen = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))
info = ti.Vector.field(n=3, dtype=ti.f32, shape=seeds_info.shape[0])
info.from_numpy(seeds_info)
voronoi = proposed_solver_2D(width, height, seeds)
voronoi.solve_proposed()
voronoi.render_color(screen, info)
plt.imshow(screen.to_numpy(), interpolation='nearest')
plt.show()
# voronoi.display() # uncomment for obtaining image of Voronoi 
ti.profiler.print_kernel_profiler_info()

# runner for proposed algo with sqrt decomp
# width, height, seeds, seeds_info = 512, 512, np.array(np.random.rand(100, 2), dtype=np.float32), np.array(np.random.rand(100, 3), dtype=np.float32)
# screen = ti.Vector.field(n=3, dtype=ti.f32, shape=(width, height))
# info = ti.Vector.field(n=3, dtype=ti.f32, shape=seeds_info.shape[0])
# info.from_numpy(seeds_info)
# voronoi = proposed_solver_2D_with_sqrt_decomp(width, height, seeds)
# voronoi.solve_proposed()
# voronoi.render_color(screen, info)
# plt.imshow(screen.to_numpy(), interpolation='nearest')
# plt.show()
# voronoi.display() # uncomment for obtaining image of Voronoi 
# ti.profiler.print_kernel_profiler_info()


    





