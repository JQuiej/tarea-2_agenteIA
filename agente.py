import sys
import gymnasium
from sympy import false
import torch

# --- PARCHE DE COMPATIBILIDAD ---
sys.modules["gym"] = gymnasium 
# --------------------------------

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

print("1. Buscando el modelo en Hugging Face...")
checkpoint_path = load_from_hub(
    repo_id="sb3/ppo-LunarLander-v2",
    filename="ppo-LunarLander-v2.zip",
)
print(f"Modelo listo en: {checkpoint_path}")

print("2. Iniciando entorno de juego con ALTA DIFICULTAD...")

# Configuración de dificultad
# enable_wind: Activa el viento.
# wind_power: Fuerza del viento (15.0 es bastante fuerte, default es 0).
# turbulence_power: Qué tan caótico es el viento (1.5 crea ráfagas impredecibles).
dificultad = {
    "render_mode": "human",
    "enable_wind": True,       
    "wind_power": 15.0,       
    "turbulence_power": 1.5   
}

try:
    # Intentamos cargar la versión moderna
    env = gymnasium.make("LunarLander-v3", **dificultad)
except gymnasium.error.NameNotFound:
    # Si falla, usamos la versión antigua con la misma dificultad
    print("Advertencia: v3 no encontrado, usando v2...")
    env = gymnasium.make("LunarLander-v2", apply_api_compatibility=True, **dificultad)


print("3. Cargando y adaptando el modelo...")

# Forzamos compatibilidad
custom_objects = {
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

model = PPO.load(checkpoint_path, env=env, custom_objects=custom_objects)

print("\n--- ¡MODO EXTREMO ACTIVADO! ---")
print("El agente ahora debe luchar contra el viento lateral.")
print("Observa cómo usa los motores laterales más agresivamente.")

try:
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Reiniciamos el juego cuando termina
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nJuego terminado.")
    env.close()
except Exception as e:
    print(f"\nOcurrió un error inesperado: {e}")
    env.close()