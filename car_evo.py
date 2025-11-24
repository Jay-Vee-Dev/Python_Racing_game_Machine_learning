import pygame
import torch
import numpy as np
import math
from tracks.track import Track
from entities.car import Car
from models.net import ActorCritic
from ga.PPO_utils import PPOBuffer, ppo_update, LR

# --- Hyperparameters ---
NUM_CARS = 20
OBS_DIM = 7  # [d1,d2,d3,speed,heading_to_finish,dist_to_finish,progress]
MAX_STEPS = 800
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Observation function ---
def get_obs(car: Car, track: Track):
    sensors = car.sensors(track.surface)
    d1, d2, d3 = sensors[0], sensors[1], sensors[2]

    # heading to finish
    finish_vec = np.array(track.finish_line[1]) - np.array(track.finish_line[0])
    finish_dir = finish_vec / (np.linalg.norm(finish_vec) + 1e-8)
    car_vec = np.array([math.cos(math.radians(car.angle)), -math.sin(math.radians(car.angle))])
    heading_reward = np.clip(np.dot(car_vec, finish_dir), -1.0, 1.0)

    # distance to finish
    car_pos = np.array([car.x, car.y])
    finish_pos = np.mean(np.array(track.finish_line), axis=0)
    dist_to_finish = np.linalg.norm(finish_pos - car_pos) / 800.0

    # progress along track
    centerline = np.array(track.centerline)
    progress = 1.0 - np.min(np.linalg.norm(centerline - car_pos, axis=1)) / 800.0

    # normalized speed
    speed = car.speed / 6.0

    obs_vec = torch.tensor([d1, d2, d3, speed, heading_reward, dist_to_finish, progress],
                           dtype=torch.float32, device=DEVICE)
    return obs_vec


# --- Initialize cars on start line ---
def init_cars(track):
    cars = []
    start_p1, start_p2 = track.start_line
    for i in range(NUM_CARS):
        t = i / (NUM_CARS - 1)  # spread along the line
        x = start_p1[0] + (start_p2[0] - start_p1[0]) * t
        y = start_p1[1] + (start_p2[1] - start_p1[1]) * t
        cars.append(Car(track=track, color_idx=i, x=x, y=y, angle=track.start_dir))
    return cars


# --- Main ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("PyCarNN - Learning Cars")
    clock = pygame.time.Clock()

    track = Track()
    buffer = PPOBuffer()
    agent = ActorCritic().to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    cars = init_cars(track)

    step = 0
    generation = 1
    best_score = 0.0

    running = True
    while running:
        obs_batch = []
        for car in cars:
            obs_batch.append(get_obs(car, track))
        obs_batch = torch.stack(obs_batch).to(DEVICE)

        # --- Forward pass ---
        with torch.no_grad():
            actions_batch, values_batch = agent(obs_batch)
        actions_batch = actions_batch.cpu()
        values_batch = values_batch.cpu()

        # --- Step cars ---
        rewards = []
        dones = []
        gen_best = -float("inf")
        for i, car in enumerate(cars):
            steer = float(actions_batch[i, 0].detach())
            throttle = float(actions_batch[i, 1].detach())
            # force initial movement if network outputs near zero
            if abs(throttle) < 0.05:
                throttle = 0.3
            car.step(steer, throttle, track.surface, finish_line=track.finish_line)

            # reward
            reward = 0.0
            if car.finished:
                reward += 1000.0
            elif not car.alive:
                reward -= 100.0
            else:
                reward += car.speed * 2.0
                reward += values_batch[i].item()

            rewards.append(reward)
            dones.append(not car.alive or car.finished)

            # store in buffer
            buffer.store(obs_batch[i], actions_batch[i],
                         torch.tensor([0.0], device=DEVICE),  # placeholder log_prob
                         reward, dones[-1], values_batch[i])

            if reward > gen_best:
                gen_best = reward

        # --- Render ---
        screen.fill((30, 30, 30))
        screen.blit(track.surface, (0, 0))
        for car in cars:
            if car.alive or car.finished:
                car.draw(screen)

        # Display generation & best score
        font = pygame.font.SysFont("Arial", 20)
        text_gen = font.render(f"Generation: {generation}", True, (255, 255, 255))
        text_best = font.render(f"Best Score: {gen_best:.1f}", True, (255, 255, 0))
        screen.blit(text_gen, (10, 10))
        screen.blit(text_best, (10, 40))

        pygame.display.flip()
        clock.tick(60)
        step += 1

        if step >= MAX_STEPS or all(dones):
            # update PPO
            ppo_update(agent, buffer, optimizer)
            buffer.clear()
            # respawn cars
            cars = init_cars(track)
            step = 0
            generation += 1
            best_score = gen_best
            print(f"Generation {generation} complete. Best Score: {best_score:.1f}")

        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()
