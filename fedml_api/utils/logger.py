import os
import json
import time
import platform
import logging


class Logger:
    _instance = None

    def __new__(cls, logger=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logger
        return cls._instance

    @classmethod
    def initialize(cls, logger):
        if cls._instance is None:
            cls._instance = Logger(logger)
        return cls._instance

    def get_logger(self):
        if self.logger is not None:
            return self.logger
        else:
            import logging

            logger = logging.getLogger("default_logger")
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
            print("WARNING: logger is not initialized. Create a default logger.")
            return logger


# def logging_config(args, process_id):
#     # customize the log format
#     while logging.getLogger().handlers:
#         logging.getLogger().handlers.clear()
#     console = logging.StreamHandler()
#     if args.level == 'INFO':
#         console.setLevel(logging.INFO)
#     elif args.level == 'DEBUG':
#         console.setLevel(logging.DEBUG)
#     else:
#         raise NotImplementedError
#     formatter = logging.Formatter(str(process_id) +
#         ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
#     console.setFormatter(formatter)
#     # Create an instance
#     logging.getLogger().addHandler(console)
#     # logging.getLogger().info("test")
#     logging.basicConfig()
#     logger = logging.getLogger()
#     if args.level == 'INFO':
#         logger.setLevel(logging.INFO)
#     elif args.level == 'DEBUG':
#         logger.setLevel(logging.DEBUG)
#     else:
#         raise NotImplementedError
#     logging.info(args)
