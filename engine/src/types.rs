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
    #[serde(rename = "batchSize")]
    pub batch_size: i64,
    pub epoch: Epoch,
    pub kl: Kl,
    pub mode: String,

    #[serde(rename = "saveInterval")]
    pub save_interval: i64,
    pub shuffle: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Kl {
    pub kl_cutoff: f64,
    pub log: bool,

    #[serde(rename = "useEpochLimit")]
    pub use_epoch_limit: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Epoch {
    pub epochs: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tetris {
    pub decay: Decay,

    #[serde(rename = "discouragedActions")]
    pub discouraged_actions: DiscouragedActions,

    #[serde(rename = "encouragedActions")]
    pub encouraged_actions: EncouragedActions,
    pub states: States,
    pub truncate: Truncate,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Truncate {
    #[serde(rename = "piecePlacementTruncates")]
    pub piece_placement_truncates: bool,

    #[serde(rename = "placementTimer")]
    pub placement_timer: PlacementTimer,

    #[serde(rename = "rewardBoundary")]
    pub reward_boundary: bool,

    #[serde(rename = "stopCollection")]
    pub stop_collection: bool,
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

    #[serde(rename = "earlyMove")]
    pub early_move: EarlyMove,
    pub edges: Edges,

    #[serde(rename = "gameOver")]
    pub game_over: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edges {
    pub enabled: bool,
    pub reward: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EarlyMove {
    #[serde(rename = "actionsReward")]
    pub actions_reward: Vec<ActionsRewardItem>,
    pub cutoff: i64,

    #[serde(rename = "diminishFactor")]
    pub diminish_factor: f64,
    pub enabled: bool,
    pub punishment: Punishment,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Punishment {
    pub factor: f64,

    #[serde(rename = "punishLateMoves")]
    pub punish_late_moves: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActionsRewardItem {
    pub name: String,
    pub reward: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cyclic {
    pub enabled: bool,

    #[serde(rename = "maxRotates")]
    pub max_rotates: i64,
    pub reward: i64,

    #[serde(rename = "rotateHorizon")]
    pub rotate_horizon: i64,
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
    #[serde(rename = "actionsUntilDrop")]
    pub actions_until_drop: i64,
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
    pub init_lr: InitLr,
    pub min_lr: MinLr,
    pub mode: String,
    pub ppo: Ppo,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ppo {
    pub entropy: f64,
    pub lamda: i64,

    #[serde(rename = "minEntropy")]
    pub min_entropy: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinLr {
    #[serde(rename = "actorLearnRate")]
    pub actor_learn_rate: f64,

    #[serde(rename = "convLearnRate")]
    pub conv_learn_rate: f64,

    #[serde(rename = "criticLearnRate")]
    pub critic_learn_rate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InitLr {
    #[serde(rename = "actorLearnRate")]
    pub actor_learn_rate: f64,

    #[serde(rename = "convLearnRate")]
    pub conv_learn_rate: f64,

    #[serde(rename = "criticLearnRate")]
    pub critic_learn_rate: f64,
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

    #[serde(rename = "maxExperiencesPerTrajectory")]
    pub max_experiences_per_trajectory: i64,
    pub mode: String,

    #[serde(rename = "parallelEnvs")]
    pub parallel_envs: i64,
    pub runs: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Erm {
    pub enabled: bool,
    pub length: i64,

    #[serde(rename = "minTrajectories")]
    pub min_trajectories: i64,
}
