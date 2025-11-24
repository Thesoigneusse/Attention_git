import logging

def setup_logger():
    logger = logging.getLogger("TRACE")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler("execution.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

if __name__ == "__main__":
    # Fichier principal
    setup_logger()  # Configure les handlers une fois pour toutes
    # apply_tracing(globals())

    # Permet de récupérer le logger dans un autre fichier
    logger = logging.getLogger("TRACE")
    logger.info(f"Exemple de logger dans un sous-fichier")