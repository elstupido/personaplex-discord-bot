import logging
import sys
import os
import warnings
from dotenv import load_dotenv

_root_configured = False

def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized, centralized logger for the bot and all third-party libs."""
    global _root_configured
    load_dotenv()
    
    log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    if not _root_configured:
        # Route all Python warnings (e.g. from torch, diffusers) into the logging system
        logging.captureWarnings(True)
        # Suppress ONLY the three specific annoying deprecation warnings requested
        warnings.filterwarnings("ignore", message=".*ignore\\(True\\) has been deprecated.*")
        warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear is deprecated.*")
        warnings.filterwarnings("ignore", message=".*weight_norm is deprecated in favor of.*")
        
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear any existing root handlers (prevents duplicates if called multiple times)
        root_logger.handlers.clear()

        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[90m',    # Gray
                'INFO': '\033[92m',     # Green
                'WARNING': '\033[93m',  # Yellow
                'ERROR': '\033[91m',    # Red
                'CRITICAL': '\033[95m'  # Magenta
            }
            RESET = '\033[0m'

            def format(self, record):
                color = self.COLORS.get(record.levelname, self.RESET)
                # Ensure the name string is correctly formatted before printing
                name_str = f"{record.name[:15]:<15}" if len(record.name) <= 15 else f"{record.name[:12]}..."
                
                record.levelname = f"{color}{record.levelname:8}{self.RESET}"
                record.msg = f"{color}{record.msg}{self.RESET}"
                
                # Replace the original format args so we don't mutate the global state
                original_name = record.name
                record.name = f"\033[94m{name_str}\033[0m"
                formatted = super().format(record)
                record.name = original_name
                return formatted

        formatter = ColoredFormatter(
            '\033[96m%(asctime)s\033[0m | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Hijack existing stubborn loggers (like Uvicorn) to force them into our format
        for logger_name, logger_obj in logging.root.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                logger_obj.handlers.clear()
                logger_obj.propagate = True

        # Suppress noisy DEBUG output from third-party libraries UNLESS we are in DEBUG mode
        if log_level > logging.DEBUG:
            for noisy in ('discord', 'aiohttp', 'websockets', 'asyncio', 'uvicorn.access'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
                
        _root_configured = True

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger
