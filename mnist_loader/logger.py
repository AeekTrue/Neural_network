import logging

LOADER_FORMAT = '%(asctime)s --- %(name)s %(levelname)s: %(msg)s'
loader_formatter = logging.Formatter(LOADER_FORMAT, datefmt='%d.%m.%Y %X')

loader_handler = logging.StreamHandler()
loader_handler.setLevel(logging.INFO)
loader_handler.setFormatter(loader_formatter)

loader_logger = logging.getLogger('MNIST loader')
loader_logger.setLevel(logging.INFO)
loader_logger.handlers = [loader_handler]
