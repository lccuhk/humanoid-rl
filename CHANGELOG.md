# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow for Python testing and linting
- Project preview SVG image
- GitHub badges (stars, forks, issues, license, last-commit)
- Tech stack badges (Python, PyTorch, MuJoCo, PyBullet, etc.)
- MIT LICENSE file

### Changed
- Added project preview screenshot section to README

## [1.2.0] - 2026-06-15

### Added
- Detailed joint logging training script
- Real-time simulation with health reward and fall penalty parameters
- Health reward optimization training script
- Auto-comparison script after training completion
- Slow-motion video recording with detailed flight phase analysis
- Training comparison script for walk vs run tasks

### Fixed
- Reward function design flaw that allowed model to gain rewards by falling
- Syntax errors in MuJoCo environment scripts

## [1.1.0] - 2026-06-01

### Added
- Optimized reward function for running tasks
- Video recording scripts for training visualization
- Optimized training scripts for faster convergence
- Detailed logging during training
- PPO verification scripts

### Changed
- Improved reward function to encourage forward movement
- Enhanced training stability with better hyperparameters

## [1.0.0] - 2026-05-15

### Added
- Initial implementation with PPO, SAC, TD3 algorithms
- MuJoCo physics simulation environment
- PyBullet physics simulation environment
- Isaac Gym environment support
- Humanoid robot model with full articulation
- Training visualization with TensorBoard
- Robot visualization with matplotlib
- Data collection and preprocessing modules
- Web UI for real-time monitoring

[Unreleased]: https://github.com/lccuhk/humanoid-rl/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/lccuhk/humanoid-rl/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/lccuhk/humanoid-rl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/lccuhk/humanoid-rl/releases/tag/v1.0.0
