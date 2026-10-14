from envs.cluster import ClusterPackEnv
from envs.sqlite_tune import SQLiteTuneEnv
from envs.serving_sim import ServingBatchEnv

ENVS = {
    "cluster": ClusterPackEnv,
    "sqlite": SQLiteTuneEnv,
    "serving": ServingBatchEnv,
}
