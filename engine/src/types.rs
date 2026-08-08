use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NNConfig {
    pub collection: Collection,
    pub network: Network,
    pub scheduler: Scheduler,
    pub tetris: Tetris,
    pub training: Training,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Training {
    pub batchSize: i64,
    pub epoch: Epoch,
    pub kl: Kl,
    pub mode: String,
    pub saveInterval: i64,
    pub shuffle: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Kl {
    pub kl_cutoff: f64,
    pub log: bool,
    pub useEpochLimit: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Epoch {
    pub epochs: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tetris {
    pub decay: Decay,
    pub discouragedActions: DiscouragedActions,
    pub encouragedActions: EncouragedActions,
    pub states: States,
    pub truncate: Truncate,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Truncate {
    pub piecePlacementTruncates: bool,
    pub placementTimer: PlacementTimer,
    pub rewardBoundary: bool,
    pub stopCollection: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlacementTimer {
    pub enabled: bool,
    pub reward: i64,
    pub value: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct States {
    pub cyclic: Cyclic,
    pub earlyMove: EarlyMove,
    pub edges: Edges,
    pub gameOver: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edges {
    pub enabled: bool,
    pub reward: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EarlyMove {
    pub actionsReward: Vec<ActionsRewardItem>,
    pub cutoff: i64,
    pub diminishFactor: f64,
    pub enabled: bool,
    pub punishment: Punishment,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Punishment {
    pub factor: f64,
    pub punishLateMoves: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActionsRewardItem {
    pub name: String,
    pub reward: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cyclic {
    pub enabled: bool,
    pub maxRotates: i64,
    pub reward: i64,
    pub rotateHorizon: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EncouragedActions {
    pub actions: Vec<String>,
    pub reward: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscouragedActions {
    pub actions: Vec<serde_json::Value>,
    pub reward: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Decay {
    pub actionsUntilDrop: i64,
    pub enabled: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Scheduler {
    pub decay: Vec<DecayItem>,
    pub patience: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecayItem {
    pub factor: f64,
    pub init: String,
    pub min: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Network {
    pub apa: Apa,
    pub clipping: Clipping,
    pub dqn: Dqn,
    pub dropout: f64,
    pub gamma: f64,
    pub init_lr: Init_lr,
    pub min_lr: Min_lr,
    pub mode: String,
    pub ppo: Ppo,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ppo {
    pub entropy: f64,
    pub lamda: i64,
    pub minEntropy: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Min_lr {
    pub actorLearnRate: f64,
    pub convLearnRate: f64,
    pub criticLearnRate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Init_lr {
    pub actorLearnRate: f64,
    pub convLearnRate: f64,
    pub criticLearnRate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dqn {
    pub temperature: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Clipping {
    pub actor_epsilon: f64,
    pub critic_epsilon: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Apa {
    pub entropy: f64,
    pub lamda: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Collection {
    pub erm: Erm,
    pub experiences: i64,
    pub maxExperiencesPerTrajectory: i64,
    pub mode: String,
    pub parallelEnvs: i64,
    pub runs: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Erm {
    pub enabled: bool,
    pub length: i64,
    pub minTrajectories: i64,
}

