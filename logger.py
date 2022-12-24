import os
import sys
import logging


class NoOp:
    """
    No operation logger.
    """
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op


def get_logger(log_dir, log_name=None, resume="", is_rank0=True):
    """
    Get the program logger.
    """
    if is_rank0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

        # FileHandler
        mode = "w+" if resume == "False" else "a+"
        if log_name is None:
            log_name = os.path.basename(sys.argv[0]).split(".")[0] + (".log")
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode=mode)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
    else:
        logger = NoOp()

    return logger