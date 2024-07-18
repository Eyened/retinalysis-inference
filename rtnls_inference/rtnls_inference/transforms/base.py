class TestTransform:
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def undo(self, batch, logits):
        return logits
