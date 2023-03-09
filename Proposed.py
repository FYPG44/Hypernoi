import numpy as np
import taichi as ti
import taichi_glsl as ts
from matplotlib import pyplot as plt
ti.init()


@ti.data_oriented
class proposed_solver_2D:
    def __init__(self, width, height, sites):
        self.w = width
        self.h = height
        self.num_site = sites.shape[0]
        self.pixels = ti.Vector.field(n=sites.shape[0], dtype=ti.i32, shape=(self.w, self.h))
        self.pixels.fill(10**9)
        self.result = ti.field(dtype=ti.i32, shape=(self.w, self.h))
        self.result.fill(-1)
        self.sites = ti.Vector.field(n=sites.shape[1], dtype=ti.f32, shape=(sites.shape[0],))
        self.sites.from_numpy(sites)
        self.num_site_sqr = self.num_site * self.num_site
        self.radii = ti.field(dtype=ti.i32, shape=(self.num_site))
    
    # repeating lines can be avoided here
    @ti.func
    def check(self, xc, yc, x, y):
        return ((xc+ x >= 0 and yc + y >= 0 and xc + x < self.w and yc + y < self.h) == True)

    # repeating lines can be avoided here        
    @ti.func
    def draw_circle(self, site_ind: ti.i32, r: ti.i32, x: ti.i32, y: ti.i32):
        index = ti.cast(
                ts.vec(self.sites[site_ind].x*self.w, self.sites[site_ind].y*self.h), ti.i32)
        xc, yc = index.x, index.y
        if (self.check(xc, yc, x, y)):
            self.pixels[xc + x, yc + y][site_ind] = r
        if (self.check(xc, yc, x, -y)):
            self.pixels[xc + x, yc - y][site_ind] = r
        if (self.check(xc, yc, -x, y)):
            self.pixels[xc - x, yc + y][site_ind] = r
        if (self.check(xc, yc, -x, -y)):
            self.pixels[xc - x, yc - y][site_ind] = r
        if (self.check(xc, yc, y, x)):
            self.pixels[xc + y, yc + x][site_ind] = r
        if (self.check(xc, yc, y, -x)):
            self.pixels[xc + y, yc - x][site_ind] = r
        if (self.check(xc, yc, -y, x)):
            self.pixels[xc - y, yc + x][site_ind] = r
        if (self.check(xc, yc, -y, -x)):
            self.pixels[xc - y, yc - x][site_ind] = r

    @ti.func
    def circle_dcs(self, r, site_ind):
        i = s = 0
        j = r
        w = r-1
        l = w << 1
        g = 1
        while i <= j:
            while True:
                self.draw_circle(site_ind, r, i, j)
                s = s + i
                i += 1
                s = s + i
                if s > w:
                    break
            if (s > w and s <= (w+g) and i <= j):
                self.draw_circle(site_ind, r, i, j)
            w = w + l
            l = l - 2
            j -= 1
            g += 2

    @ti.func
    def grow_circle(self, site_ind):
        for r in range(self.radii[site_ind]):
            self.circle_dcs(r, site_ind)

    @ti.kernel
    def fill_frames(self):
        for i in range(self.num_site):
            self.grow_circle(i)
    
    @ti.func
    def find_color(self, i, j):
        radius, seed = 10**9, -1
        for k in range(self.num_site):
            if self.pixels[i, j][k] < radius:
                radius = self.pixels[i, j][k]
                seed = k
        return seed
        
    @ti.kernel
    def generate_result(self):
        for i, j in self.result:
            self.result[i, j] = self.find_color(i, j)

    @ti.kernel
    def render_color(self, screen: ti.template(), site_info: ti.template()):
        for I in ti.grouped(screen):
            if self.result[I] != -1:
                screen[I] = site_info[self.result[I]]
            else:
                screen[I].fill(-1)
    @ti.kernel
    def render_index(self, screen: ti.template()):
        for I in ti.grouped(screen):
            if self.result[I] != -1:
                screen[I].fill(self.result[I] / self.num_site)
            else:
                screen[I].fill(-1)
                    
    # @ti.kernel
    def display(self):
        plt.imshow(self.result.to_numpy(), interpolation='nearest')
        plt.show()
    

    @ti.func
    def find_avg_radii(self, i, j):
        index = ti.cast(
                ts.vec(self.sites[i].x*self.w, self.sites[i].y*self.h), ti.i32)
        xc, yc = index.x, index.y

        index = ti.cast(
                ts.vec(self.sites[j].x*self.w, self.sites[j].y*self.h), ti.i32)
        xc1, yc1 = index.x, index.y
        self.radii[i] += max(abs(xc - xc1), abs(yc - yc1))



    @ti.kernel
    def find_optimum_radii(self):
        for num in range(1, self.num_site_sqr+1):
            j = num % self.num_site
            if j == 0:
                j = self.num_site
            i = (num + self.num_site - 1) // self.num_site

            self.find_avg_radii(i-1, j-1)


    @ti.func
    def find_avg_radii_final(self, i):
        self.radii[i] //= 2*self.num_site
        print(self.radii[i])
        # self.radii[i] = 100


    @ti.kernel
    def find_optimum_radii_final(self):
        for i in range(self.num_site):
            self.find_avg_radii_final(i)


    # @ti.func
    def solve_proposed(self):
        self.find_optimum_radii()
        self.find_optimum_radii_final()
        self.fill_frames()
        self.generate_result()