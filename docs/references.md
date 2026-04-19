# Control references

The paper or book each implementation follows. Each module also cites it in its docstring.

| Component | `control_mode` / module | Reference |
| --------- | ----------------------- | --------- |
| Pure Pursuit | `pure_pursuit` · `control/controllers.py` | R. C. Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm", CMU-RI-TR-92-01, 1992 |
| Stanley | `stanley` · `control/controllers.py` | S. Thrun et al., "Stanley: The Robot that Won the DARPA Grand Challenge", *J. Field Robotics* 23(9), 2006; G. Hoffmann et al., "Autonomous Automobile Trajectory Tracking for Off-Road Driving", ACC 2007 |
| LQR (path-error model) | `lqr` · `control/lqr.py` | J. M. Snider, "Automatic Steering Methods for Autonomous Automobile Path Tracking", CMU-RI-TR-09-08, 2009; R. Rajamani, *Vehicle Dynamics and Control*, Springer, 2012 |
| Linear time-varying MPC | `mpc` · `control/mpc.py` | J. Kong, M. Pfeiffer, G. Schildbach, F. Borrelli, "Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design", IEEE IV 2015; P. Falcone et al., "Predictive Active Steering Control for Autonomous Vehicle Systems", *IEEE TCST* 15(3), 2007 |
| QP solver | `control/mpc.py` | B. Stellato et al., "OSQP: an operator splitting solver for quadratic programs", *Math. Prog. Comp.*, 2020 |
| Follow the Gap | `depth_ftg` · `control/depth_ftg.py`; `ftg` · `workers/track_ftg_controller.py` | V. Sezer, M. Gokasan, "A novel obstacle avoidance algorithm: 'Follow the Gap Method'", *Robotics and Autonomous Systems* 60(9), 2012 |
| VAE control model (World Models V + C) | `vae` · `control/vae.py`, `scripts/train_vae_policy.py` | D. Ha, J. Schmidhuber, "World Models", arXiv:1803.10122, 2018; D. P. Kingma, M. Welling, "Auto-Encoding Variational Bayes", ICLR 2014; I. Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017 |
| Behaviour cloning (end-to-end CNN) | `bc` · `control/bc.py`, `scripts/train_bc_policy.py` | M. Bojarski et al., "End to End Learning for Self-Driving Cars" (PilotNet), arXiv:1604.07316, 2016; D. A. Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network", NIPS 1989 |
| Noise-injected imitation (BC / VAE training data) | `control/imitation.py` | M. Laskey, J. Lee, R. Fox, A. Dragan, K. Goldberg, "DART: Noise Injection for Robust Imitation Learning", CoRL 2017 |
| PPO policy | `policy` · `control/policy.py`, `rl/` | J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017 |
| Speed profile | `control/speed.py` | N. R. Kapania, J. Subosits, J. C. Gerdes, "A Sequential Two-Step Algorithm for Fast Generation of Vehicle Racing Trajectories", *J. Dyn. Sys., Meas., Control* 138(9), 2016 |
| PID (speed, legacy steering) | `control/speed.py`, `pid` | K. J. Åström, R. M. Murray, *Feedback Systems*, Princeton University Press, 2008 |
| Extended Kalman filter | `control/estimator.py` | S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005 |
| Kinematic bicycle model | `control/sim.py`, `control/lqr.py` | R. Rajamani, *Vehicle Dynamics and Control*, Springer, 2012; Kong et al., IEEE IV 2015 |

Implementation notes that go beyond the papers:
- **Follow the Gap:** gap centre blended with the goal angle, as in the paper. The best gap is
  scored by visible free area rather than width, so a nearby obstacle face can't win.
  The virtual scan comes from the depth image with a floor/height filter.
- **MPC:** condensed onto the steering inputs, with steering-angle and steering-rate
  constraints and the actuation delay propagated before solving.
- **LQR:** gains scheduled on speed with a curvature feed-forward `atan(L·κ)`.
- **BC:** PilotNet layout scaled to the 40×64 depth input, trained end to end on steering;
  same demonstrations, follower and cruise-speed logic as the VAE model (`control/imitation.py`).
- **VAE:** trained on depth frames rather than RGB, since depth transfers from the
  bench to the car better than rendered colour. The head is trained on the frozen latent
  means (the World Models V/C separation). Frames are mirrored to balance turns.
  Speed follows a cruise setting that drops with steering.
- **Speed control:** there is no brake. Neutral coasts, and a limited backward input is used
  above 0.3 m/s only.
