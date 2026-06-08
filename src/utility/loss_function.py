from neuralop.losses import LpLoss

class ComplexLpLoss(LpLoss):
    """
    Wrapper for the LpLoss implemented in the 'neuralop' package to allow the use of complex losses.
    """
    def __init__(self, d=1, p=1, measure=1.0, reduction="sum", eps=1e-8, take_root=False):
        self.take_root=take_root
        super().__init__(d, p, measure, reduction, eps)

    def __call__(self, y_pred, y, **kwargs):
        real_error = super().rel(y_pred.real, y.real, take_root=self.take_root)
        imag_err = super().rel(y_pred.imag, y.imag, take_root=self.take_root)
        return real_error + imag_err