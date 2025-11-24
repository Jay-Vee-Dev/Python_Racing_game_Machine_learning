import math
import pygame
import random

class Car:
    PREDEFINED_COLORS = [
        (0, 255, 255),    # neon blue
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (255, 255, 0),    # yellow
        (255, 0, 255),    # magenta
        (0, 128, 255),    # bright sky blue
        (255, 128, 0),    # orange
        (128, 0, 255),    # purple
        (0, 255, 128),    # mint green
        (255, 192, 203),  # pink
    ]

    def __init__(self, x=None, y=None, angle=None, color=None, track=None, color_idx=None):
        if track and track.centerline:
            self.x, self.y = track.centerline[0] if x is None or y is None else (x, y)
            self.angle = track.start_dir if angle is None else angle
        else:
            self.x, self.y = (400, 100) if x is None or y is None else (x, y)
            self.angle = 0.0 if angle is None else angle

        self.speed = 0.0
        self.alive = True
        self.distance = 0.0
        self.time_alive = 0
        self.finished = False

        # Assign color
        if color is not None:
            self.color = color
        else:
            if color_idx is not None and color_idx < len(self.PREDEFINED_COLORS):
                self.color = self.PREDEFINED_COLORS[color_idx]
            else:
                self.color = random.choice(self.PREDEFINED_COLORS)

    def sensors(self, track_mask):
        readings = []
        for deg in (-45, 0, 45):
            angle = math.radians(self.angle + deg)
            dist = 0
            for d in range(1, 120, 3):
                sx = int(self.x + math.cos(angle) * d)
                sy = int(self.y - math.sin(angle) * d)
                if sx < 0 or sy < 0 or sx >= track_mask.get_width() or sy >= track_mask.get_height():
                    dist = d
                    break
                if track_mask.get_at((sx, sy)).a == 0:
                    dist = d
                    break
            readings.append(dist / 120.0)
        return readings

    def step(self, steer, throttle, track_mask, finish_line=None):
        if not self.alive or self.finished:
            return  # Crash or finished

        # Finish line check
        if finish_line:
            x1, y1 = finish_line[0]
            x2, y2 = finish_line[1]
            dx = x2 - x1
            dy = y2 - y1
            if dx != 0 or dy != 0:
                t = ((self.x - x1) * dx + (self.y - y1) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                closest_x = x1 + t*dx
                closest_y = y1 + t*dy
                dist = math.hypot(self.x - closest_x, self.y - closest_y)
                if dist < 6:
                    self.finished = True
                    self.speed = 0
                    return

        # Movement
        max_turn = 2.0
        steer = max(-1.0, min(1.0, steer))
        self.angle += steer * max_turn
        self.angle *= 0.98  # damping to reduce spinning

        self.speed += throttle * 0.3
        self.speed = max(0.0, min(6.0, self.speed))

        dx = math.cos(math.radians(self.angle)) * self.speed
        dy = -math.sin(math.radians(self.angle)) * self.speed
        self.x += dx
        self.y += dy
        self.distance += abs(dx) + abs(dy)
        self.time_alive += 1

        px, py = int(self.x), int(self.y)
        if px <= 0 or py <= 0 or px >= track_mask.get_width() or py >= track_mask.get_height():
            self.alive = False
            return
        if track_mask.get_at((px, py)).a == 0:
            self.alive = False

    def draw(self, surface):
        car_surf = pygame.Surface((12, 20), pygame.SRCALPHA)
        pygame.draw.polygon(car_surf, self.color, [(6, 0), (12, 20), (0, 20)])
        car_surf = pygame.transform.rotate(car_surf, self.angle)
        surface.blit(car_surf, car_surf.get_rect(center=(self.x, self.y)))
