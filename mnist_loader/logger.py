from loguru import logger
import sys
logger.remove(0)
FORMAT = '[{time:HH:mm:ss}] <lvl>{name} {level}: {message}</>'
logger.add(sys.stderr, level='INFO', format=FORMAT, filter=lambda rec: rec['name'] == 'mnist_loader')