helpers: with helpers; {
    PATIENCE = 100;
    MIN_LEARN_RATE = e- 6;
    DECAY_RATE = 0.1;
    ENTROPY_DECAY_AMOUNT = 0.01;
    ENTROPY_MIN = 0.001;

    network = {
        type = "ppo";
        lr = {
            convLearnRate = e- 5;
            actorLearnRate = e- 5;
            criticLearnRate = e- 5;
        };
        dropout = 0.2;
        gamma = 0.9;
        ppo = {
            lambda = 0.1;
            entropy = 0.1;
            clipEpsilon = 0.2;
        };
        dqn = {
            temperature = 1.1;
        };
    };
    training = {
        type = "epoch";
        batchSize = 64;
        saveInterval = 5;
        shuffle = false;
        epoch = {
            epochs = 10;
        };
    };
    collection = {
        experiences = {
            parallelEnvs = 1;
            maxExperiencesPerTrajectory = 100;
        };
        runs = 100;
    };
    tetris = {
        decay = {
            enabled = true;
            actionsUntilDrop = 10;
        };
        truncate = {
            rewardBoundary = true;
            stopCollection = true;
            piecePlacementTruncates = true;
            placementTimer = {
                enabled = true;
                value = 50;
                reward = -10;
            };
        };
        encouragedActions = {
            actions = ["HARD_DROP" "DOWN"];
            reward = 1;
        };
        discouragedActions = {
            actions = [];
            reward = -0.1;
        };
        states = {
            gameOver = -100;
            cyclic = {
                enabled = true;
                maxRotates = 4;
                reward = -1;
            };
            edges = {
                enabled = true;
                reward = -1;
            };
        };
    };
}