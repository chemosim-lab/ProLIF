import logging

logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("prolif")
logger.setLevel(logging.INFO)
