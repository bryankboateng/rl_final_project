import yaml


def convert_config(new_path='config_new.yaml', out_path='config.yaml'):
    with open(new_path, 'r') as f:
        new_cfg = yaml.safe_load(f)

    old_cfg = {}

    # === ARCHITECTURE ===
    arch = new_cfg.get("arch", {})
    old_cfg["arch"] = {
        "CRITIC_HAS_ACT_IND": False,
        "ACTIVATION": {
            "actor": arch.get("actor_0", {}).get("activation", "Sin"),
            "critic": arch.get("critic_0", {}).get("activation", "Sin"),
        },
        "APPEND_DIM": arch.get("actor_0", {}).get("append_dim", 0),
        "LATENT_DIM": arch.get("actor_0", {}).get("latent_dim", 0),
        "DIM_LIST": {
            "actor_0": arch.get("actor_0", {}).get("mlp_dim", []),
            "actor_1": arch.get("actor_1", {}).get("mlp_dim", []),
            "critic": arch.get("critic_0", {}).get("mlp_dim", []),
        },
        "ACTION_RANGE": [
            arch.get("actor_0", {}).get("action_range", []),
            arch.get("actor_1", {}).get("action_range", []),
        ],
        "ACTION_DIM": [
            arch.get("actor_0", {}).get("action_dim", 0),
            arch.get("actor_1", {}).get("action_dim", 0)
        ],
        "OBS_DIM": {
            "actor_0": arch.get("actor_0", {}).get("obsrv_dim", 0),
            "actor_1": arch.get("actor_1", {}).get("obsrv_dim", 0),
            "critic": arch.get("critic_0", {}).get("obsrv_dim", 0)
        }
    }

    # === ENVIRONMENT ===
    env = new_cfg.get("environment", {})
    old_cfg["environment"] = {
        "SEED": env.get("seed", 0),
        "NUM_AGENTS": 2,
        "TIMEOUT": env.get("timeout", 500),
        "END_CRITERION": env.get("end_criterion", "failure"),
    }

    # === AGENT ===
    agent = new_cfg.get("agent", {})
    force_info = agent.get("force_info") or {}
    old_cfg["agent"] = {
        "DYN": agent.get("dyn", "Unknown"),
        "FOOTPRINT": agent.get("footprint", "none"),
        "VERBOSE": agent.get("verbose", False),
        "GUI": agent.get("gui", False),
        "GUI_IMAGINARY": agent.get("gui_imaginary", False),
        "DT": agent.get("dt", 0.02),
        "APPLY_FORCE": agent.get("apply_force", False),
        "REPLACE_ADV_WITH_DR": agent.get("replace_adv_with_dr", False),
        "FORCE": agent.get("force", 0),
        "FORCE_SCALE": agent.get("force_scale", 1.0),
        "FORCE_RESET_TIME": agent.get("force_reset_time", 0),
        "FORCE_INFO": force_info,
        "LINK_NAME": force_info.get("link_name", ""),
        "ROTATE_RESET": agent.get("rotate_reset", True),
        "HEIGHT_RESET": agent.get("height_reset", "both"),
        "FORCE_RANDOM": True,
        "TERRAIN": agent.get("terrain", "normal"),
        "TERRAIN_HEIGHT": agent.get("terrain_height", 0.1),
        "TERRAIN_GRIDSIZE": agent.get("terrain_gridsize", 0.2),
        "TERRAIN_FRICTION": agent.get("terrain_friction", 1.0),
        "ENVTYPE": agent.get("envtype", "normal"),
        "ACTION_RANGE": {
            "CTRL": agent.get("action_range", {}).get("ctrl", []),
            "DSTB": agent.get("action_range", {}).get("dstb", [])
        },
        "NUM_SEGMENT": 1,
        "AGENT_ID": agent.get("agent_id", "ego"),
        "PRETRAIN_CTRL": "",
        "PRETRAIN_DSTB": "",
        "RESET_CRITERION": agent.get("reset_criterion", "failure")
    }

    # === SOLVER ===
    solver = new_cfg.get("solver", {})
    eval_cfg = solver.get("eval", {})

    old_cfg["solver"] = {
        "USE_WANDB": solver.get("use_wandb", True),
        "PROJECT_NAME": solver.get("project_name", "SMART"),
        "NAME": solver.get("name", "go2_isaacs_debug"),
        "OUT_FOLDER": solver.get("out_folder", "train_result/smart/go2_isaacs"),
        "CHECK_OPT_FREQ": 20,
        "SAVE_TOP_K": [50, 5],
        "NUM_CPUS": solver.get("num_envs", 1),
        "MAX_STEPS": solver.get("max_steps", 8_000_000),
        "MEMORY_CAPACITY": solver.get("memory_capacity", 1_000_000),
        "MIN_STEPS_B4_OPT": solver.get("min_steps_b4_opt", 100_000),
        "OPTIMIZE_FREQ": solver.get("opt_period", 10_000),
        "UPDATE_PER_OPT": [2000, solver.get("num_updates_per_opt", 1000)],
        "CTRL_OPT_FREQ": solver.get("ctrl_update_ratio", 10),
        "MIN_STEPS_B4_EXPLOIT": 0,
        "NUM_EVAL_TRAJ": eval_cfg.get("num_trajectories", 20),
        "EVAL_TIMEOUT": eval_cfg.get("timeout", 500),
        "NUM_ENVS": solver.get("num_envs", 1),
        "WARMUP_ACTION_RANGE": {
            "CTRL": solver.get("warmup_action_range", {}).get("ctrl", []),
            "DSTB": solver.get("warmup_action_range", {}).get("dstb", [])
        },
        "ROLLOUT_END_CRITERION": solver.get("rollout_end_criterion", "reach-avoid"),
        "VENV_DEVICE": solver.get("device", "cpu"),
        "HISTORY_WEIGHT": 0.0,
        "DSTB_SAMPLE_TYPE": "softmax",
        "INIT_DSTB_SAMPLE_TYPE": "strongest",
        "DSTB_SAMPLE_CUR_WEIGHT": 0.2,
        "CHECK_NOM": False
    }

    # === UPDATE ===
    critic_0 = solver.get("critic_0", {})
    actor_0 = solver.get("actor_0", {})
    actor_1 = solver.get("actor_1", {})

    old_cfg["update"] = {
        "MAX_MODEL": solver.get("max_model", 50),
        "ALPHA": [actor_0.get("alpha", 0.1), actor_1.get("alpha", 0.1)],
        "LEARN_ALPHA": actor_0.get("learn_alpha", True),
        "BATCH_SIZE": solver.get("batch_size", 256),
        "DEVICE": solver.get("device", "cpu"),
        "OPT_TYPE": actor_0.get("opt_type", "AdamW"),
        "GAMMA": critic_0.get("gamma", 0.9),
        "GAMMA_DECAY": critic_0.get("gamma_decay", 0.1),
        "GAMMA_END": critic_0.get("gamma_end", 0.999),
        "GAMMA_PERIOD": critic_0.get("gamma_period", 1_000_000),
        "GAMMA_SCHEDULE": critic_0.get("gamma_schedule", True),
        "LATENT_DIM": 0,
        "LR_A": actor_0.get("lr", 0.0001),
        "LR_C": critic_0.get("lr", 0.0001),
        "LR_Al": [actor_0.get("lr_al", 0.000125), actor_1.get("lr_al", 0.0000125)],
        "LR_A_END": actor_0.get("lr_end", 0.0001),
        "LR_C_END": critic_0.get("lr_end", 0.0001),
        "LR_Al_END": [actor_0.get("lr_al_end", 0.00005), actor_1.get("lr_al_end", 0.000005)],
        "LR_A_PERIOD": actor_0.get("lr_period", 50000),
        "LR_C_PERIOD": critic_0.get("lr_period", 50000),
        "LR_Al_PERIOD": [actor_0.get("lr_al_period", 100000), actor_1.get("lr_al_period", 100000)],
        "LR_A_DECAY": actor_0.get("lr_decay", 0.9),
        "LR_C_DECAY": critic_0.get("lr_decay", 0.9),
        "LR_Al_DECAY": [actor_0.get("lr_al_decay", 0.9), actor_1.get("lr_al_decay", 0.9)],
        "LR_A_SCHEDULE": actor_0.get("lr_schedule", False),
        "LR_C_SCHEDULE": critic_0.get("lr_schedule", False),
        "LR_Al_SCHEDULE": [actor_0.get("lr_al_schedule", False), actor_1.get("lr_al_schedule", False)],
        "MODE": critic_0.get("mode", "reach-avoid"),
        "TAU": critic_0.get("tau", 0.01),
        "TERMINAL_TYPE": critic_0.get("terminal_type", "max"),
        "EVAL": critic_0.get("eval", False),
        "UPDATE_PERIOD": [actor_0.get("update_period", 2), actor_1.get("update_period", 2)],
        "ACTOR_TYPE": [actor_0.get("actor_type", "max"), actor_1.get("actor_type", "min")]
    }

    # === EVAL ===
    eval_top = new_cfg.get("eval", {})
    old_cfg["eval"] = {
        "MODEL_TYPE": eval_top.get("model_type", ["manual", "manual"]),
        "STEP": eval_top.get("step", [0, 0]),
        "EVAL_TIMEOUT": eval_top.get("eval_timeout", 1000),
        "IMAGINARY_HORIZON": eval_top.get("imaginary_horizon", 500),
    }

    # === Save ===
    with open(out_path, "w") as f:
        yaml.dump(old_cfg, f, sort_keys=False)

    print(f"[✓] Converted config saved to {out_path}")


if __name__ == "__main__":
    convert_config()
