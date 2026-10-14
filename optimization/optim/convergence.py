"""Conservative first-order stopping, independent of optimizer learning rate."""
from collections import deque
import math
import torch


class ConvergenceMonitor:
    def __init__(self, step_tol, loss_rtol=1e-5, grad_tol=1e-3, window=5):
        self.step_tol, self.loss_rtol, self.grad_tol = step_tol, loss_rtol, grad_tol
        self.values = deque(maxlen=window)
        self.phase = None

    def reset(self):
        self.values.clear()
        self.phase = None

    def observe(self, loss, step, gradient, phase=None):
        if not all(math.isfinite(x) for x in (loss, step, gradient)):
            raise FloatingPointError('Nonfinite optimization loss, update or gradient')
        if phase != self.phase:
            self.values.clear()
            self.phase = phase
        self.values.append((loss, step, gradient))
        if len(self.values) < self.values.maxlen:
            return 'optimizing'
        losses, steps, grads = zip(*self.values)
        small_step = max(steps) < self.step_tol
        stable_loss = max(losses) - min(losses) <= self.loss_rtol * max(1., abs(losses[0]))
        if small_step and stable_loss and max(grads) < self.grad_tol:
            return 'converged'
        return 'stalled' if small_step else 'optimizing'


def optimizer_parameters(optimizers):
    seen = set()
    result = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if id(parameter) not in seen:
                    result.append(parameter)
                    seen.add(id(parameter))
    return result


def optimizer_snapshot(optimizers):
    parameters = optimizer_parameters(optimizers)
    gradients = [p.grad.detach().abs().max().item() for p in parameters if p.grad is not None]
    # No gradient is not a stationarity certificate.
    return [p.detach().clone() for p in parameters], max(gradients, default=float('inf'))


def optimizer_step_norm(optimizers, before):
    values = [(p.detach() - old).square().sum() for p, old in zip(optimizer_parameters(optimizers), before)]
    return float(torch.stack(values).sum().sqrt())
