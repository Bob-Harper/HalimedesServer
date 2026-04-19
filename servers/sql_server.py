import logging

logger = logging.getLogger("SQLServer")

# ------------------------------------------------------------
# Placeholder server file for symmetry with other servers.
# This project uses the system-level SQL server directly.
# No local SQL server process is started or managed here.
# ------------------------------------------------------------

logger.info("[SQLServer] Placeholder module loaded (no local SQL server).")

if __name__ == "__main__":
    print(
        "\n[HalServerGateway]\n"
        "This module is part of the Hal service stack.\n"
        "It is NOT meant to be executed directly.\n"
        "Start Hal using:\n"
        "    start_hal_services.py\n"
    )
    import sys
    sys.exit(0)