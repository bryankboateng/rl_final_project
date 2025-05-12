# RL Project


---

````markdown
#  Reproducing and Stress Testing Safety-Critical Policies with ISAACS and Gameplay Filters

This repository enables training safety filters and conducting Bespoke Ultimate Stress Tests (BUST) for evaluating control robustness in simulated quadrupedal locomotion environments. We build on top of the ISAACS framework and integrate adversarial best-response policies and gameplay safety filters.

````

###  Train a Safety Controller (ISAACS)

```bash
# Inside config/isaacs_spirit/, set:
#   - out_folder
#   - (Optional) wandb.project and wandb.name
python3 train_isaacs.py -cf config/isaacs_spirit.yaml
```

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




##  Running BUST Evaluation

The BUST (Bespoke Ultimate Stress Test) evaluation consists of **three main steps**:

---

###  Step 1: Generate Initial Conditions

Create a pool of initial states that are nominally safe when executing the controller without adversaries:

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

>  Not all generated states are valid (some may hang or be unstable). We **validate** these in the next step.

---

### Step 2: Validate Batch in Chunks (Sanity Check)

Before full evaluation, ensure initial states are valid for a given control-disturbance pair:

```bash
python3 script/eval_rarl_batch.py \
  --ctrl_type safety \
  --ctrl_config train_result/isaacs_spirit/config.yaml \
  --ctrl_step 2500000 \
  --dstb_type adversary \
  --dstb_config train_result/test_spirit_refactor/test_bust-safety-2/config.yaml \
  --dstb_step 1300000 \
  --batch_path batch_1000_bust_1000.pkl \
  --index_range 0 99 \
  --eval_horizon 1000 \
  --log \
  --log_name bust_safety_safety_chunk0 \
  --exp_name bust_validation
```

Repeat `--index_range` for other chunks (100–199, 200–299, ...) to identify valid slices.

>  After validation, use `combine_batches.py` to merge verified subsets into one clean `.pkl` file.

---

###  Step 3: Full Evaluation on Clean Batch

Once you’ve assembled a fully valid batch, run the actual BUST metric evaluation:

```bash
python3 script/eval.py \
  --ctrl_type safety \
  --ctrl_config train_result/isaacs_spirit/config.yaml \
  --ctrl_step 2500000 \
  --dstb_type adversary \
  --dstb_config train_result/test_spirit_refactor/test_bust-safety-2/config.yaml \
  --dstb_step 1300000 \
  --batch_path batch_1000_bust_1000_clean.pkl \
  --index_range 0 999 \
  --eval_horizon 1000 \
  --log \
  --log_name bust_safety_safety_final \
  --exp_name bustnew
```

> This is the step that produces the **final safe rate** result for each safety disturbance pair.




To run **random domain randomization** baselines:

* Leave out `--dstb_type`.
* Inside the `dstb_config`:

  * Set `replace_adv_with_dr: true`
  * Set `force_type: uniform` (for regular random) or `bangbang` (for extreme-valued).

---

##  Output

Each evaluation logs the **safe rate** — the percentage of runs that do not end in failure. These statistics form the basis for the BUST performance table across control–disturbance pairings.

---

##  Code Attribution

###  Contributions:

* `utils_implemented.py`: helper utilities
* `actors_and_critics.py`:

  * `TwinnedQNetwork`, `GaussianPolicy`
  * Actor & critic `update()` and critic `value()` implementations with other marked features identical to homework solutions and handouts.
* Training and testing networks for Spirit, Go2 quadruped and pybullet humanoid.
* Full PyBullet humanoid wrapper (written and tested but not used in final experiments)

###  Integration:

* Merged codebase from:

  * [gameplay-filter repo](https://github.com/SafeRoboticsLab/Gameplay-Filters/tree/develop) #public
  * [safe\_adaptation\_dev repo](https://github.com/SafeRoboticsLab/safe_adaptation_dev/tree/gameplay-release) #Note this repo provided needed backbone for BUST training: requires permission from Princeton SafeRobotics Lab
* Reconciled naming, API compatibility, and rollout pipelines for training and evaluating with dynamics.
* Repos served as references for difficult aspects of implementation.


---

Video Links for GO2 Demos

Edited Video:
https://drive.google.com/file/d/1gLsvo-I51N6GN33a277H7yl1Qcs5JBgk/view?usp=sharing

Safety Off Raw Video:
https://drive.google.com/file/d/1mbrLapKC6fo9hDSrobRyyZzUgsMs8cbh/view?usp=sharing

Safety On Raw Video:
https://drive.google.com/file/d/1XTuGgMT8o4V22QE22zVVNw1adRKry31s/view?usp=sharing


Here’s a clean `README.md`-style formatting for your acknowledgement and citation section:

---

## Acknowledgements

We extend our sincere thanks to the **SafeRobotics Lab** for providing access to both the private repository and the **Go2 quadruped platform**. This support enabled us to validate the effectiveness of our **ISAACS** pipeline through real-world deployment of our trained networks.

## Citation

If you use or reference our work or the ISAACS framework, please cite the following paper:

```bibtex
@inproceedings{hsunguyen2023isaacs,
  title     = {ISAACS: Iterative Soft Adversarial Actor-Critic for Safety},
  author    = {Kai-Chieh Hsu and Duy P. Nguyen and Jaime F. Fisac},
  booktitle = {Proceedings of the 5th Conference on Learning for Dynamics and Control},
  year      = {2023},
}
```

---

