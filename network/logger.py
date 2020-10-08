from loguru import logger
import sys

SGD_FORMAT = '[{time:HH:mm:ss}] <lvl>{name} {level}:</> Epoch #{extra[epoch]} {extra[right_answers_present]}%'
CONF_FORMAT = '[{time:HH:mm:ss}] <lvl>{name} {level}:</> {message}'
logger.add(sys.stderr, level='INFO', format=SGD_FORMAT, filter=lambda rec: rec['name'] == 'network')
logger.add(sys.stderr, level='INFO', format=CONF_FORMAT, filter=lambda rec: rec['name'] == 'network.configurator')
