from dataclasses import dataclass
@dataclass
class NNConfig:
	collection: Collection
	network: Network
	scheduler: Scheduler
	tetris: Tetris
	training: Training

@dataclass
class Training:
	batchSize: int
	epoch: Epoch
	kl: Kl
	mode: str
	saveInterval: int
	shuffle: bool

@dataclass
class Kl:
	kl_cutoff: float
	log: bool
	useEpochLimit: bool

@dataclass
class Epoch:
	epochs: int

@dataclass
class Tetris:
	decay: Decay
	discouragedActions: DiscouragedActions
	encouragedActions: EncouragedActions
	states: States
	truncate: Truncate

@dataclass
class Truncate:
	piecePlacementTruncates: bool
	placementTimer: PlacementTimer
	rewardBoundary: bool
	stopCollection: bool

@dataclass
class PlacementTimer:
	enabled: bool
	reward: int
	value: int

@dataclass
class States:
	cyclic: Cyclic
	earlyMove: EarlyMove
	edges: Edges
	gameOver: int

@dataclass
class Edges:
	enabled: bool
	reward: int

@dataclass
class EarlyMove:
	actionsReward: list
	cutoff: int
	diminishFactor: float
	enabled: bool
	punishment: Punishment

@dataclass
class Punishment:
	factor: float
	punishLateMoves: bool

@dataclass
class Cyclic:
	enabled: bool
	maxRotates: int
	reward: int
	rotateHorizon: int

@dataclass
class EncouragedActions:
	actions: list
	reward: int

@dataclass
class DiscouragedActions:
	actions: list
	reward: float

@dataclass
class Decay:
	actionsUntilDrop: int
	enabled: bool

@dataclass
class Scheduler:
	decay: list
	patience: int

@dataclass
class Network:
	apa: Apa
	clipping: Clipping
	dqn: Dqn
	dropout: float
	gamma: float
	init_lr: Init_lr
	min_lr: Min_lr
	mode: str
	ppo: Ppo

@dataclass
class Ppo:
	entropy: float
	lamda: int
	minEntropy: float

@dataclass
class Min_lr:
	actorLearnRate: float
	convLearnRate: float
	criticLearnRate: float

@dataclass
class Init_lr:
	actorLearnRate: float
	convLearnRate: float
	criticLearnRate: float

@dataclass
class Dqn:
	temperature: float

@dataclass
class Clipping:
	actor_epsilon: float
	critic_epsilon: float

@dataclass
class Apa:
	entropy: float
	lamda: int

@dataclass
class Collection:
	erm: Erm
	experiences: int
	maxExperiencesPerTrajectory: int
	mode: str
	parallelEnvs: int
	runs: int

@dataclass
class Erm:
	enabled: bool
	length: int
	minTrajectories: int
