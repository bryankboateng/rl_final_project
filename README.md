# RL Project


---

````markdown
#  Reproducing and Stress Testing Safety-Critical Policies with ISAACS and Gameplay Filters

This repository enables training safety filters and conducting *Bespoke Ultimate Stress Tests (BUST)* for evaluating control robustness in simulated quadrupedal locomotion environments. We build on top of the ISAACS framework and integrate adversarial best-response policies and gameplay safety filters.

---

## Training Policies

### Train a Safety Controller (ISAACS)

```bash
# Inside config/isaacs_spirit/, set:
#   - out_folder
#   - (Optional) wandb.project and wandb.name

python3 train_isaacs.py -cf config/isaacs_spirit.yaml
````

###  Train a Go2 Controller

```bash
# Inside config/isaacs_go2/, set:
#   - out_folder
#   - (Optional) wandb.project and wandb.name

python3 train_isaacs.py -cf config/isaacs_go2.yaml
```

###  Train Adversarial Best-Response Disturbance Policies

Train disturbances against fixed control policies:

```bash
python3 train_l2.py -cf config/spirit_bust_task.yaml       # Task-only controller
python3 train_l2.py -cf config/spirit_bust_safety.yaml     # Safety controller
python3 train_l2.py -cf config/spirit_bust_value.yaml      # Value-based safety
python3 train_l2.py -cf config/spirit_bust_gameplay.yaml   # Gameplay safety filter
```

Notes:

* Set `out_folder` and specify the control checkpoint to freeze.
* For gameplay, tune `gameplay_horizon` carefully — too short leads to poor attacks, too long slows training.
* `epsilon` only matters for `shield_value`.

---

## Running BUST Evaluation

### Step 1: Generate Initial Conditions

```bash
python3 script/batch_generate_init_conditions.py \
  -cf train_result/isaacs_spirit/config.yaml \
  --runs 1000 \
  --type stay_safe \
  --ctrl_step 2500000 \
  --dstb_step 2400000 \
  --reset_criterion failure \
  --end_criterion failure \
  --imaginary_horizon 300 \
  --suffix bust_1000
```

>  Some generated states may hang or be invalid. Run evaluation in chunks (e.g., 0–99, 100–199), validate each, then merge clean subsets using `combine_batches.py`.

---

### Step 2: Run BUST Evaluation for Each Pair

```bash
python3 script/eval_rarl_batch.py \
  --ctrl_type safety \
  --ctrl_config train_result/isaacs_spirit/config.yaml \
  --ctrl_step 2500000 \
  --dstb_type adversary \
  --dstb_config train_result/test_spirit_refactor/test_bust-safety-2/config.yaml \
  --dstb_step 1300000 \
  --batch_path batch_1000_bust_1000.pkl \
  --index_range 0 999 \
  --eval_horizon 1000 \
  --log \
  --log_name bust_safety_safety \
  --exp_name bustnew
```

---

### Step 3: Final Evaluation on Full Clean Batch (Example)

```bash
python3 script/eval.py \
  --ctrl_type safety \
  --ctrl_config train_result/isaacs_spirit/config.yaml \
  --ctrl_step 2500000 \
  --dstb_type adversary \
  --dstb_config train_result/test_spirit_refactor/test_bust-safety-2/config.yaml \
  --dstb_step 1300000 \
  --batch_path batch_1000_bust_1000.pkl \
  --index_range 0 999 \
  --eval_horizon 1000 \
  --log \
  --log_name bust_safety_safety \
  --exp_name bustnew
```

To run **random domain randomization** baselines:

* Leave out `--dstb_type`.
* Inside the `dstb_config`:

  * Set `replace_adv_with_dr: true`
  * Set `force_type: uniform` (for regular random) or `bangbang` (for extreme-valued).

---

##  Output

Each evaluation logs the **safe rate** — the percentage of runs that do not end in failure. These statistics form the basis for the BUST performance table across control–disturbance pairings.

---

## 💡 Code Attribution

###  Contributions:

* `utils/functions.py`: helper utilities
* `actors_and_critics.py`:

  * `TwinnedQNetwork`, `GaussianPolicy`
  * Actor & critic `update()` and critic `value()` implementations
* Full PyBullet humanoid wrapper (written but not used in final experiments)

###  Integration:

* Merged codebase from:

  * [gameplay-filter repo](https://github.com/SafeRoboticsLab/semantic_gameplay_filters)
  * [safe\_adaptation\_dev repo](https://github.com/SafeRoboticsLab/safe_adaptation_dev)
* Reconciled naming, API compatibility, and rollout pipelines

---

