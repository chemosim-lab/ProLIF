import logging

# get logger
logger = logging.getLogger()
# set logger to debug
logger.setLevel(logging.DEBUG)
# Add logs to the terminal (stderr)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
