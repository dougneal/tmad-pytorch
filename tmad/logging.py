import logging


def configure_logging(level_name):
    level = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }.get(level_name)

    log_format = (
        '%(asctime)s | %(levelname)s | '
        '%(name)s %(funcName)s | %(message)s'
    )
    logging.basicConfig(format=log_format, level=level)
