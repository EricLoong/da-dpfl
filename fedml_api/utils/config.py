class Config:
    _instance = None

    def __new__(cls, args=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.args = args
        return cls._instance

    @classmethod
    def initialize(cls, args):
        if cls._instance is None:
            cls._instance = Config(args)
        return cls._instance

    def get_args(self):
        return self.args
