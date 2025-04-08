import torch.optim as optim

class AdaSmoothDelta(optim.Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, beta=0.9):
        defaults = dict(lr=lr, rho=rho, eps=eps, beta=beta)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["acc_grad"] = torch.zeros_like(p.data)
                    state["acc_delta"] = torch.zeros_like(p.data)
                    state["smoothed_grad"] = torch.zeros_like(p.data)

                acc_grad = state["acc_grad"]
                acc_delta = state["acc_delta"]
                smoothed_grad = state["smoothed_grad"]

                # Сглаженный градиент
                smoothed_grad.mul_(group["beta"]).add_(grad, alpha=1 - group["beta"])

                acc_grad.mul_(group["rho"]).addcmul_(smoothed_grad, smoothed_grad, value=1 - group["rho"])

                update = (acc_delta + group["eps"]).sqrt() / (acc_grad + group["eps"]).sqrt() * smoothed_grad

                p.data.add_(update, alpha=-group["lr"])

                acc_delta.mul_(group["rho"]).addcmul_(update, update, value=1 - group["rho"])
