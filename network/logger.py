import logging

SGD_FORMAT = '%(asctime)s --- %(name)s %(levelname)s: Epoch #%(epoch)s %(right_answers_present)3f%%'
sgd_formatter = logging.Formatter(SGD_FORMAT, datefmt='%d.%m.%Y %X')

sgd_handler = logging.StreamHandler()
sgd_handler.setLevel(logging.INFO)
sgd_handler.setFormatter(sgd_formatter)

sgd_logger = logging.getLogger('SGD')
sgd_logger.setLevel(logging.INFO)
sgd_logger.handlers = [sgd_handler]


CONFIGURATOR_FORMAT = '%(asctime)s --- %(name)s %(levelname)s: %(msg)s'
configurator_formatter = logging.Formatter(CONFIGURATOR_FORMAT, datefmt='%d.%m.%Y %X')

configurator_handler = logging.StreamHandler()
configurator_handler.setLevel(logging.INFO)
configurator_handler.setFormatter(configurator_formatter)

configurator_logger = logging.getLogger('Configurator')
configurator_logger.setLevel(logging.INFO)
configurator_logger.handlers = [configurator_handler]
