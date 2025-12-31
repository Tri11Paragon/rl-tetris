from collections import namedtuple

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

PPOExperience = namedtuple("PPOExperience", ["state", "action", "reward", "done", "logprob", "state_value"])
PPOExperience2 = namedtuple("PPOExperience2",
                            ["state", "action", "reward", "done", "logprob", "state_value", "state_metric"])
