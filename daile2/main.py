import numpy as np
from scipy.integrate import solve_ivp
import pygame
from collections import deque

# --- Fizikiniai ir Simuliacijos Parametrai ---
L1 = 1.0  # Pirmojo strypo ilgis (m)
L2 = 1.0  # Antrojo strypo ilgis (m)
M1 = 1.0  # Pirmosios masės dydis (kg)
M2 = 1.0  # Antrosios masės dydis (kg)
G = 9.81  # Laisvojo kritimo pagreitis (m/s^2)

# Pradinės sąlygos: [theta1, omega1, theta2, omega2]
# theta1, theta2: kampai radianais (nuo vertikalios ašies žemyn)
# omega1, omega2: kampiniai greičiai (rad/s)
THETA1_INIT = np.pi * 0.9 # Pvz., šiek tiek pakreipta
OMEGA1_INIT = 0.0
THETA2_INIT = np.pi * 1.1 # Pvz., į kitą pusę
OMEGA2_INIT = 0.0
initial_conditions = [THETA1_INIT, OMEGA1_INIT, THETA2_INIT, OMEGA2_INIT]

# Simuliacijos laikas
T_MAX = 60.0  # Bendras simuliacijos laikas (s)
FPS_ANIMATION = 60 # Norimas animacijos kadrų skaičius per sekundę
N_POINTS_SIM = int(T_MAX * FPS_ANIMATION) # Taškų skaičius sprendinyje
t_eval = np.linspace(0, T_MAX, N_POINTS_SIM)

# --- Vizualizacijos Parametrai ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 800

# Švytuoklės pakabinimo taškas
ORIGIN_X = SCREEN_WIDTH // 2
ORIGIN_Y = SCREEN_HEIGHT // 3

# Mastelis: pritaikome, kad švytuoklė tilptų ekrane
# Maksimalus ilgis L1+L2. Tarkim, norim, kad tai užimtų apie ORIGIN_Y erdvės.
# Arba fiksuotas dalis ekrano aukščio, pvz., 1/3.
# Apskaičiuojame taip, kad L1+L2 būtų maždaug (SCREEN_HEIGHT - ORIGIN_Y) * 0.8 arba ORIGIN_Y * 0.8
# Pasirinkime mažesnį iš galimų verčių, kad tilptų visomis kryptimis.
max_pendulum_reach_pixels = min(ORIGIN_Y, SCREEN_HEIGHT - ORIGIN_Y, ORIGIN_X, SCREEN_WIDTH - ORIGIN_X) * 0.9
SCALE = max_pendulum_reach_pixels / (L1 + L2) # pikseliai vienam metrui

L1_PX = int(L1 * SCALE)
L2_PX = int(L2 * SCALE)

BOB_RADIUS1 = int(M1** (1/3) * 8) # Dydis priklauso nuo masės (kubinė šaknis)
BOB_RADIUS2 = int(M2** (1/3) * 8)
ARM_THICKNESS = 4
TRACE_LEN = 1000  # Pėdsako taškų ilgis
TRACE_THICKNESS = 2

# Spalvos
COLOR_BACKGROUND = (20, 20, 30) # Tamsiai mėlyna/pilka
COLOR_ARM1 = (200, 200, 200)
COLOR_ARM2 = (180, 180, 180)
COLOR_BOB1 = (255, 100, 100) # Raudona
COLOR_BOB2 = (100, 100, 255) # Mėlyna
COLOR_TRACE = (100, 255, 100, 150) # Žalia su šiek tiek permatomumo (alpha)

# --- Diferencialinės lygtys dvigubai švytuoklei ---
def double_pendulum_derivs(t, y, L1, L2, m1, m2, g):
    theta1, omega1, theta2, omega2 = y

    # Tarpiniai skaičiavimai, kad lygtys būtų aiškesnės
    delta_theta = theta1 - theta2
    cos_delta = np.cos(delta_theta)
    sin_delta = np.sin(delta_theta)

    # Denominators (Bendri vardikliai abiem pagreičiams, bet šiek tiek skiriasi)
    den1 = (m1 + m2) * L1 - m2 * L1 * cos_delta * cos_delta
    den2 = (L2/L1) * den1 # Supaprastinimui, jei L1 yra vardiklyje

    # Kampinis pagreitis pirmajai masei (alpha1 arba theta1_ddot)
    alpha1 = (m2 * L1 * omega1 * omega1 * sin_delta * cos_delta +
              m2 * g * np.sin(theta2) * cos_delta -
              m2 * L2 * omega2 * omega2 * sin_delta -
              (m1 + m2) * g * np.sin(theta1)) / den1

    # Kampinis pagreitis antrajai masei (alpha2 arba theta2_ddot)
    # Atkreipkite dėmesį, kad šios lygtys yra viena iš galimų formų.
    # Svarbu naudoti nuoseklią ir teisingą formuluotę.
    # Naudojama formuluotė iš https://www.myphysicslab.com/pendulum/double-pendulum-en.html
    # (pritaikyta kintamiesiems)

    alpha1_num = -g * (2 * m1 + m2) * np.sin(theta1) \
                 - m2 * g * np.sin(theta1 - 2 * theta2) \
                 - 2 * sin_delta * m2 * (omega2**2 * L2 + omega1**2 * L1 * cos_delta)
    alpha1_den = L1 * (2 * m1 + m2 - m2 * np.cos(2 * delta_theta)) # Originalus vardiklis buvo 2*(theta1-theta2)
    alpha1 = alpha1_num / alpha1_den

    alpha2_num = 2 * sin_delta * (omega1**2 * L1 * (m1 + m2) +
                                  g * (m1 + m2) * np.cos(theta1) +
                                  omega2**2 * L2 * m2 * cos_delta)
    alpha2_den = L2 * (2 * m1 + m2 - m2 * np.cos(2 * delta_theta)) # Originalus vardiklis buvo 2*(theta1-theta2)
    alpha2 = alpha2_num / alpha2_den


    return [omega1, alpha1, omega2, alpha2]

# --- Pagrindinė programa ---
def main():
    # Sprendžiame diferencialines lygtis
    print("Skaičiuojama švytuoklės trajektorija...")
    sol = solve_ivp(
        double_pendulum_derivs,
        (0, T_MAX),
        initial_conditions,
        args=(L1, L2, M1, M2, G),
        dense_output=True, # Reikalinga, kad galėtume gauti sprendinius t_eval taškuose
        t_eval=t_eval,
        method='RK45' # Galima bandyti 'LSODA' jei problemos su standumu
    )
    print("Skaičiavimas baigtas.")

    # Pygame inicializacija
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Dviguba Švytuoklė (Patobulinta)")
    clock = pygame.time.Clock()

    # Pėdsako taškų sąrašas (deque efektyvesnis)
    trace_points = deque(maxlen=TRACE_LEN)
    
    # Sukuriame paviršių pėdsakui su alpha kanalu (permatomumui)
    trace_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    trace_surface.fill((0,0,0,0)) # Pilnai permatomas pradžioje

    running = True
    frame_index = 0
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r: # Reset - iš naujo paleisti simuliaciją
                    frame_index = 0
                    trace_points.clear()
                    trace_surface.fill((0,0,0,0)) # Išvalyti pėdsako paviršių

        if not paused:
            if frame_index < len(sol.t):
                # Gauname švytuoklės būseną dabartiniam kadrui
                theta1_val = sol.y[0, frame_index]
                omega1_val = sol.y[1, frame_index] # Nenaudojama piešimui, bet yra
                theta2_val = sol.y[2, frame_index]
                omega2_val = sol.y[3, frame_index] # Nenaudojama piešimui, bet yra

                # Konvertuojame kampus į Dekarto koordinates
                x1 = ORIGIN_X + L1_PX * np.sin(theta1_val)
                y1 = ORIGIN_Y + L1_PX * np.cos(theta1_val)
                x2 = x1 + L2_PX * np.sin(theta2_val)
                y2 = y1 + L2_PX * np.cos(theta2_val)

                # Pridedame antrosios masės poziciją į pėdsaką
                if len(trace_points) > 0: # Piešiame liniją nuo paskutinio taško
                    pygame.draw.line(trace_surface, COLOR_TRACE, trace_points[-1], (int(x2), int(y2)), TRACE_THICKNESS)
                
                trace_points.append((int(x2), int(y2)))
                
                # Jei pėdsakas per ilgas, seniausi taškai automatiškai išmetami dėl deque maxlen
                # Bet jei norime "išblukinti" seną pėdsaką ant trace_surface:
                if frame_index % 2 == 0: # Ne kiekvieną kadrą, kad būtų greičiau
                    trace_surface.set_alpha(250) # Šiek tiek sumažiname viso paviršiaus alpha kas kelis kadrus
                                                # Tai sukurs lėto išblukimo efektą.
                                                # Alternatyva: kas kartą perpiešti trace_points su mažėjančiu alpha.
                                                # Arba naudoti atskirą "blukimo" paviršių.
                                                # Paprastesnis variantas - tiesiog piešti ant pagrindinio ekrano be išblukimo.
                                                # Šiuo atveju, trace_surface kaups pėdsaką.

                frame_index += 1
            else:
                # Simuliacija baigėsi, galima sustoti arba cikliškai kartoti
                # paused = True # Sustabdyti automatiškai pabaigoje
                pass # Tiesiog nieko nedaryti, jei norime, kad stovėtų pabaigoje

        # Piešimas
        screen.fill(COLOR_BACKGROUND)

        # Piešiame pėdsaką (visą sukauptą ant trace_surface)
        screen.blit(trace_surface, (0,0))
        
        # Piešiame dabartinę pėdsako liniją (jei norime ryškesnės naujausios dalies)
        # Jei trace_points pakankamai, galima piešti pygame.draw.lines(screen, COLOR_TRACE, False, list(trace_points), TRACE_THICKNESS)
        # Bet tai dubliuotųsi su trace_surface. Paliekam trace_surface pagrindiniam pėdsakui.

        # Atnaujiname koordinates, jei simuliacija sustabdyta, bet norime matyti paskutinę būseną
        if frame_index > 0:
            current_frame_to_draw = min(frame_index, len(sol.t) -1) # Kad neviršytų ribų
            theta1_val = sol.y[0, current_frame_to_draw]
            theta2_val = sol.y[2, current_frame_to_draw]
            x1 = ORIGIN_X + L1_PX * np.sin(theta1_val)
            y1 = ORIGIN_Y + L1_PX * np.cos(theta1_val)
            x2 = x1 + L2_PX * np.sin(theta2_val)
            y2 = y1 + L2_PX * np.cos(theta2_val)

            # Piešiame švytuoklės strypus (naudojam anti-aliased, jei norim gražiau, bet storesnėms linijoms paprastos geriau)
            pygame.draw.line(screen, COLOR_ARM1, (ORIGIN_X, ORIGIN_Y), (int(x1), int(y1)), ARM_THICKNESS)
            pygame.draw.line(screen, COLOR_ARM2, (int(x1), int(y1)), (int(x2), int(y2)), ARM_THICKNESS)

            # Piešiame mases
            pygame.draw.circle(screen, COLOR_BOB1, (int(x1), int(y1)), BOB_RADIUS1)
            pygame.draw.circle(screen, COLOR_BOB2, (int(x2), int(y2)), BOB_RADIUS2)
            # Anti-aliased apskritimai (jei norima)
            # pygame.gfxdraw.aacircle(screen, int(x1), int(y1), BOB_RADIUS1, COLOR_BOB1)
            # pygame.gfxdraw.filled_circle(screen, int(x1), int(y1), BOB_RADIUS1, COLOR_BOB1)
            # pygame.gfxdraw.aacircle(screen, int(x2), int(y2), BOB_RADIUS2, COLOR_BOB2)
            # pygame.gfxdraw.filled_circle(screen, int(x2), int(y2), BOB_RADIUS2, COLOR_BOB2)


        pygame.display.flip()
        clock.tick(FPS_ANIMATION)

    pygame.quit()

if __name__ == '__main__':
    main()
