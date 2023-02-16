import numpy as np
import taichi as ti
import taichi_glsl as ts
ti.init()
from matplotlib import pyplot as plt

@ti.data_oriented
class jfa_solver_2D:
    def __init__(self, width, height, sites):
        self.w = width
        self.h = height
        self.num_site = sites.shape[0]
        self.pixels = ti.field(ti.i32, shape=(self.w, self.h))
        self.sites = ti.Vector.field(n=sites.shape[1], dtype=ti.f32, shape=(sites.shape[0],))
        self.sites.from_numpy(sites)

    @ti.kernel
    def init_sites(self):
        for i, j in self.pixels:
            self.pixels[i, j] = -1
        for i in range(self.num_site):
            index = ti.cast(
                ts.vec(self.sites[i].x*self.w, self.sites[i].y*self.h), ti.i32)
            # 1+JFA
            for x, y in ti.ndrange((-1, 2), (-1, 2)):
                index_off = ts.vec(index.x + x, index.y + y)
                if 0 <= index_off.x < self.w and 0 <= index_off.y < self.h:
                    self.pixels[index_off] = i

    @ti.kernel
    def jfa_step(self, step_x: ti.i32, step_y: ti.i32):
        for i, j in self.pixels:
            min_distance = 1e10
            min_index = -1
            for x, y in ti.ndrange((-1, 2), (-1, 2)):
                ix = i+x*step_x
                jy = j+y*step_y
                if 0 <= ix < self.w and 0 <= jy < self.h:
                    if self.pixels[ix, jy] != -1:
                        dist = ts.distance(ts.vec(i/self.w, j/self.h),
                                           self.sites[self.pixels[ix, jy]])
                        if dist < min_distance:
                            min_distance = dist
                            min_index = self.pixels[ix, jy]
            self.pixels[i, j] = min_index

    def solve_jfa(self, init_step):
        self.init_sites()
        step_x = init_step[0]
        step_y = init_step[1]
        while True:
            self.jfa_step(step_x, step_y)
            step_x = step_x // 2
            step_y = step_y // 2
            if step_x == 0 and step_y == 0:
                break
            else:
                step_x = 1 if step_x < 1 else step_x
                step_y = 1 if step_y < 1 else step_y

    @ ti.kernel
    def render_color(self, screen: ti.template(), site_info: ti.template()):
        for I in ti.grouped(screen):
            if self.pixels[I] != -1:
                screen[I] = site_info[self.pixels[I]]
            else:
                screen[I].fill(-1)

    @ti.kernel
    def render_index(self, screen: ti.template()):
        for I in ti.grouped(screen):
            if self.pixels[I] != -1:
                screen[I].fill(self.pixels[I] / self.num_site)
            else:
                screen[I].fill(-1)

    def display(self):
        plt.imshow(self.pixels.to_numpy(), interpolation='nearest')
        plt.show()

   