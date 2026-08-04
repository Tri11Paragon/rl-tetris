import numpy as np
import torch

# Relative entropy or Kullback-Leibler divergence: additional amount of information (in nats)
# required to specify the value of x as a result of using q(x) instead of p(x):

def normalize_probability_distribution(
    x: torch.Tensor | np.ndarray, eps: float = 1e-8) -> torch.Tensor | np.ndarray:

    mmin = x.min()
    mmax = x.max()
    new_x = (x - mmin) / (mmax - mmin)
    total = new_x.sum().clip(min=eps)
    return new_x / total

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def discrete_kl(p, q):
    pass

def kl_monte_carlo(log_p: torch.Tensor | np.ndarray, log_q: torch.Tensor | np.ndarray):
    """
            Monte Carlo estimator of KL(P || Q).
            Samples must be drawn from P.
        """
    return -(log_p - log_q)

def kl_approx(log_p: torch.Tensor | np.ndarray, log_q: torch.Tensor | np.ndarray):
    """
        Non-negative Monte Carlo estimator of KL(P || Q).

        Samples must be drawn from P.
    """
    log_ratio = log_p - log_q
    approx_kl = ((log_ratio.exp() - 1) - log_ratio)
    return approx_kl

def test():
    import matplotlib.pyplot as plt
    import torch.distributions as dist
    size = 100000

    p = dist.Normal(loc=0, scale=1)
    q = dist.Normal(loc=1.2, scale=3)

    true_kl = dist.kl_divergence(p, q)

    x = p.sample(sample_shape=(10_000_000,))

    print("True KL", true_kl)
    print("Backwards True KL", dist.kl_divergence(q, p))
    print()

    log_p = p.log_prob(x)
    log_q = q.log_prob(x)

    print("Forward Approx KL", kl_approx(log_p, log_q).mean())
    print("Forward Monte Carlo KL", kl_monte_carlo(log_p, log_q).mean())
    print()

    print("Backward Approx KL", kl_approx(log_q, log_p).mean())
    print("Backward Monte Carlo KL", kl_monte_carlo(log_q, log_p).mean())

    # plt.bar(x, n1)
    # plt.bar(x, n2, alpha=0.5)
    # plt.legend(["n1", "n2"])
    # plt.show()

if __name__ == "__main__":
    test()