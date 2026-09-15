# Dataset split policy

Official images and targets are not part of this repository. Each developer points the commands at a separately obtained `200_train_cases` directory.

- `0001..0150`: development and local evaluation.
- `0151..0200`: sealed holdout; do not inspect, tune, train, or evaluate.
- `10_GTcase`: sealed holdout; do not inspect, tune, train, or evaluate.

Historical `v2_*.json` manifests contain machine-specific paths and are ignored. Current commands select cases by ID and never require those files.
