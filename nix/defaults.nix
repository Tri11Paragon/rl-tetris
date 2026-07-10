helpers: with helpers; {
    network = {
        # 'ppo' or 'dqn'
        type = "ppo";
        init_lr = {
            convLearnRate = e- 5;
            actorLearnRate = e- 5;
            criticLearnRate = e- 5;
        };
        min_lr = {
            convLearnRate = e- 7;
            actorLearnRate = e- 7;
            criticLearnRate = e- 7;
        };
        dropout = 0.2;
        gamma = 0.9;
        ppo = {
            lamda = 0.1;
            entropy = 0.1;
            minEntropy = 0.001;
            clipEpsilon = 0.2;
        };
        dqn = {
            temperature = 1.1;
        };
        model = {
            conv_filters = {
                modules = [
                    {type="conv2d"; i=2; o=32; kernel=[3 3]; padding=1;}
                ];
                lr = "network.init_lr.convLearnRate";
            };
        };
    };
    training = {
        type = "kl";
        batchSize = 64;
        saveInterval = 5;
        shuffle = false;
        epoch = {
            epochs = 10;
        };
        kl = {
            kl_cutoff = 0.03;
            useEpochLimit = true;
            log = false;
        };
    };
    collection = {
        # either 'runs' or 'experiences'
        type = "experiences";
        experiences = {
            parallelEnvs = 1;
            maxExperiencesPerTrajectory = 100;
        };
        erm = {
            enabled = false;
            minTrajectories = 10;
            length = 10000;
        };
        runs = 100;
        minExperiences = 2000;
    };
    scheduler = {
        decay = [
            {init = "network.init_lr"; min = "network.min_lr"; factor = 0.1;}
            {init = "network.ppo.entropy"; min="network.ppo.minEntropy"; factor=0.9;}
        ];
        patience = 100;
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
                reward = -25;
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
                rotateHorizon = 12;
                reward = -5;
            };
            edges = {
                enabled = true;
                reward = -5;
            };
            earlyMove = {
                enabled = true;
                # if an action isn't in this it doesn't reward
                actionsReward = [
                    {name = "LEFT"; reward = 0.5;}
                    {name = "RIGHT"; reward = 0.5;}
                    {name = "ROTATE"; reward = 0.1;}
                ];
                cutoff = 0;
            };
        };
    };
}