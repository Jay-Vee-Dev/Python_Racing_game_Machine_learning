import pygame
import numpy as np
from scipy.interpolate import splprep, splev

SCREEN_W, SCREEN_H = 800, 600

class Track:
    def __init__(self, width=SCREEN_W, height=SCREEN_H):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), flags=pygame.SRCALPHA)
        self.centerline = []
        self.start_line = None
        self.finish_line = None
        self.start_dir = None  # angle in degrees along the track
        self.generate()

    def generate(self, track_width=60):
        """
        Static S-shaped track with proper start/finish lines across the track width
        """
        self.surface.fill((0, 0, 0, 0))

        # Define S-shaped control points
        points = [
            [50, 500],   # bottom left start
            [200, 400],
            [400, 200],
            [600, 400],
            [750, 500]   # top right finish
        ]
        points = np.array(points)

        # Smooth the path with cubic spline
        tck, u = splprep([points[:,0], points[:,1]], s=0, per=False)
        u_fine = np.linspace(0, 1, 1500)
        x_smooth, y_smooth = splev(u_fine, tck)
        self.centerline = list(zip(x_smooth, y_smooth))

        # Draw track as filled white circles
        for x, y in self.centerline:
            pygame.draw.circle(self.surface, (255, 255, 255, 255), (int(x), int(y)), track_width//2)

        # Compute perpendicular vector for start/finish lines
        def perp_vector(p1, p2, length):
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            norm = (dx**2 + dy**2)**0.5
            dx /= norm
            dy /= norm
            # Perpendicular
            return (-dy*length/2, dx*length/2)

        # Start line (green)
        start_pt = self.centerline[0]
        next_pt = self.centerline[5]
        perp = perp_vector(start_pt, next_pt, track_width)
        pygame.draw.line(self.surface, (0,255,0),
                         (start_pt[0]+perp[0], start_pt[1]+perp[1]),
                         (start_pt[0]-perp[0], start_pt[1]-perp[1]), 4)
        self.start_line = [(start_pt[0]+perp[0], start_pt[1]+perp[1]),
                           (start_pt[0]-perp[0], start_pt[1]-perp[1])]
        # Angle of track at start
        dx, dy = next_pt[0]-start_pt[0], next_pt[1]-start_pt[1]
        self.start_dir = np.degrees(np.arctan2(dy, dx))

        # Finish line (red)
        finish_pt = self.centerline[-1]
        prev_pt = self.centerline[-6]
        perp = perp_vector(prev_pt, finish_pt, track_width)
        pygame.draw.line(self.surface, (255,0,0),
                         (finish_pt[0]+perp[0], finish_pt[1]+perp[1]),
                         (finish_pt[0]-perp[0], finish_pt[1]-perp[1]), 4)
        self.finish_line = [(finish_pt[0]+perp[0], finish_pt[1]+perp[1]),
                            (finish_pt[0]-perp[0], finish_pt[1]-perp[1])]
