import numpy as np
import taichi as ti
import taichi_glsl as ts
from matplotlib import pyplot as plt
ti.init()


@ti.data_oriented
class proposed_solver_2D_with_sqrt_decomp:
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
        self.sqrt_site = (int)(self.num_site**0.5)
        self.minn = ti.Vector.field(n=2, dtype=ti.i32, shape=(self.w, self.h, self.sqrt_site))
        self.minn.fill(10**9)
        self.ar = self.w * self.h
        self.ar_chunk = self.ar * self.sqrt_site
    
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
        for r in range(80):
            self.circle_dcs(r, site_ind)

    @ti.kernel
    def fill_frames(self):
        for i in range(self.num_site):
            self.grow_circle(i)
    
    @ti.func
    def find_color(self, i, j, k):
        
        radius, seed = 10**9, -1
        for ind in range(k * self.sqrt_site, (k+1) * self.sqrt_site):
            if self.pixels[i, j][ind] < radius:
                radius = self.pixels[i, j][ind]
                seed = k
        self.minn[i,j,k][0]=radius
        self.minn[i,j,k][1]=seed
    

        
    @ti.kernel
    def generate_result(self):
        for num in range(1, self.ar_chunk + 1):
            k = (num + (self.ar) - 1) // (self.ar)    
            x = num % (self.ar)
            if x == 0:
                x = self.ar
            j = x % self.h
            if j == 0:
                j = self.h
            i = (x + self.h - 1) // self.h

            self.find_color(i-1,j-1,k-1)
            



    @ti.func
    def find_final_color(self, i, j):
        radius, seed = 10**9, -1
        for k in range(self.sqrt_site):
            if self.minn[i, j, k][0] < radius:
                radius = self.minn[i, j, k][0]
                seed = self.minn[i,j,k][1]
        return seed
            


    @ti.kernel
    def generate_final_result(self):
        for i, j in self.result:
            self.result[i, j] = self.find_final_color(i, j)

                    
    # @ti.kernel
    def display(self):
        plt.imshow(self.result.to_numpy(), interpolation='nearest')
        plt.show()
    
    # @ti.func
    def solve_proposed(self):
        self.fill_frames()
        self.generate_result()
        self.generate_final_result()