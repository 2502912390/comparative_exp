"""Official ACM early stopping, separate from shared strict-ACC checkpoint choice."""
import math


class OfficialEarlyStopping:
    def __init__(self,patience=100):
        self.patience=patience
        self.best_acc=0.0
        self.best_loss=math.inf
        self.curr_step=0

    def consider(self,validation_acc,validation_loss):
        if not math.isfinite(validation_acc) or not math.isfinite(validation_loss):
            raise ValueError("Nonfinite native validation stopping statistic")
        if validation_acc>=self.best_acc or validation_loss<=self.best_loss:
            self.best_acc=max(validation_acc,self.best_acc)
            self.best_loss=min(validation_loss,self.best_loss)
            self.curr_step=0
        else:self.curr_step+=1
        return self.curr_step==self.patience
