"""TF1 Adam update: epsilon in uncorrected second-moment denominator."""
import math
import torch


class TF1Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.005, betas=(0.9,0.999), eps=1e-8):
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            b1,b2=group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:continue
                gradient=parameter.grad
                if gradient.is_sparse:
                    raise ValueError("HAN trainable parameter gradients must be dense")
                state=self.state[parameter]
                if not state:
                    state["step"]=0
                    state["m"]=torch.zeros_like(parameter)
                    state["v"]=torch.zeros_like(parameter)
                state["step"]+=1
                state["m"].mul_(b1).add_(gradient,alpha=1-b1)
                state["v"].mul_(b2).addcmul_(gradient,gradient,value=1-b2)
                rate=group["lr"]*math.sqrt(1-b2**state["step"])/(1-b1**state["step"])
                denominator=state["v"].sqrt().add_(group["eps"])
                parameter.addcdiv_(state["m"],denominator,value=-rate)
        return loss
