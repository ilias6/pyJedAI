








class SharedData:
    def __init__(self, data, blocks):
        self.data = data
        self.blocks = blocks

class MultiprocessEntityMatching:
    def __init__(
            self, data, blocks, n_processes: int = 1
    ) -> any:
        self.n_processes = n_processes
        self.data = data
        self.blocks = blocks
