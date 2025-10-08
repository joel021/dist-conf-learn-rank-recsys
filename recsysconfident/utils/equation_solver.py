from scipy.stats import norm
from scipy.optimize import root_scalar


def solve_normal_pdf(mu: float, std: float, estimation: float) -> float | None:

    def equation(x):
        return x - norm.pdf(x, loc=mu, scale=std) # x - normal_pdf(x, mu, sigma) = 0

    result = estimation
    try:
        solution = root_scalar(equation, bracket=[-10*std + estimation, 10*std + estimation], method='brentq')
        if solution.converged:
            result = solution.root
    except Exception as e:
        pass

    return result

