import numpy as np
import taichi as ti
import taichi_glsl as ts
from matplotlib import pyplot as plt

# initialize Taichi
ti.init()

# define a class for the proposed solver
@ti.data_oriented
class proposed_solver_2D:
    # initialize the class with the width, height, and site locations
    def __init__(self, width, height, sites):
        self.w = width
        self.h = height
        self.num_site = sites.shape[0]
        
        # create a field to store the pixel values for each site
        self.pixels = ti.Vector.field(n=sites.shape[0], dtype=ti.i32, shape=(self.w, self.h))
        self.pixels.fill(10**9)
        
        # create a field to store the resulting site index for each pixel
        self.result = ti.field(dtype=ti.i32, shape=(self.w, self.h))
        self.result.fill(-1)
        
        # create a field to store the site locations
        self.sites = ti.Vector.field(n=sites.shape[1], dtype=ti.f32, shape=(sites.shape[0],))
        self.sites.from_numpy(sites)
    
    # check if a given pixel coordinate is within the image bounds
    @ti.func
    def check(self, xc, yc, x, y):
        return ((xc+ x >= 0 and yc + y >= 0 and xc + x < self.w and yc + y < self.h) == True)

    # draw a circle centered at a given site index and radius, with offsets x and y
    @ti.func
    def draw_circle(self, site_ind: ti.i32, r: ti.i32, x: ti.i32, y: ti.i32):
        # convert the site position from [0,1] to pixel coordinates
        index = ti.cast(
                ts.vec(self.sites[site_ind].x*self.w, self.sites[site_ind].y*self.h), ti.i32)
        xc, yc = index.x, index.y
        # if the pixel is within bounds, set its value to the circle radius
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

    # draw a circle using the DCS algorithm
    @ti.func
    def circle_dcs(self, r, site_ind):
        # This function draws a circle using the DCS algorithm
        i = s = 0
        j = r
        w = r-1
        l = w << 1
        g = 1
        while i <= j:
            while True:
                self.draw_circle(site_ind, r, i, j) # draw pixel at (i, j) with site_ind color
                s = s + i
                i += 1
                s = s + i
                if s > w:
                    break
            if (s > w and s <= (w+g) and i <= j):
                self.draw_circle(site_ind, r, i, j) # draw pixel at (i, j) with site_ind color
            w = w + l
            l = l - 2
            j -= 1
            g += 2

    @ti.func
    def grow_circle(self, site_ind):
        # This function grows a circle with increasing radius
        for r in range(80+1):
            self.circle_dcs(r, site_ind)

    @ti.kernel
    def fill_frames(self):
        # This function fills all the frames with growing circles for each site
        for i in range(self.num_site):
            self.grow_circle(i)

    @ti.func
    def find_color(self, i, j):
        # This function finds the color of the pixel at (i, j) with the smallest radius value
        radius, seed = 10**9, -1
        for k in range(self.num_site):
            if self.pixels[i, j][k] < radius:
                radius = self.pixels[i, j][k]
                seed = k
        return seed

    @ti.kernel
    def generate_result(self):
        # This function generates the result image by finding the color for each pixel
        for i, j in self.result:
            self.result[i, j] = self.find_color(i, j)

    @ti.kernel
    def render_color(self, screen: ti.template(), site_info: ti.template()):
        # This function renders the color version of the result image
        for I in ti.grouped(screen):
            if self.result[I] != -1:
                screen[I] = site_info[self.result[I]]
            else:
                screen[I].fill(-1)

    @ti.kernel
    def render_index(self, screen: ti.template()):
        # This function renders the indexed version of the result image
        for I in ti.grouped(screen):
            if self.result[I] != -1:
                screen[I].fill(self.result[I] / self.num_site)
            else:
                screen[I].fill(-1)

    @ti.kernel
    def display(self):
        # This function displays the result image using matplotlib
        plt.imshow(self.result.to_numpy(), interpolation='nearest')
        plt.show()

    # @ti.func
    def solve_proposed(self):
        # This function solves the problem by filling all the frames, generating the result image, and rendering it
        self.fill_frames()
        self.generate_result()
