import unittest
import logging
import os
import sys
import tempfile
import time
from importlib import reload

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Need to reload logger module potentially if tests modify its state
from src.utils import logger

class TestLogger(unittest.TestCase):

    def setUp(self):
        """Reset logger state before each test."""
        # Reset the internal cache of configured loggers
        logger._configured_loggers = {}
        # Create a temporary directory for log files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file_path = os.path.join(self.temp_dir.name, 'test.log')
        # Ensure default log file path is also temporary for isolation
        logger.DEFAULT_LOG_FILE = os.path.join(self.temp_dir.name, 'default_test.log')


    def tearDown(self):
        """Clean up temporary directory."""
        # Close handlers to release files before deleting directory
        for log_instance in logger._configured_loggers.values():
            for handler in log_instance.handlers[:]: # Iterate over a copy
                handler.close()
                log_instance.removeHandler(handler)
        self.temp_dir.cleanup()
        # Reset default path just in case
        logger.DEFAULT_LOG_FILE = "analysis.log"


    def test_setup_logger_defaults(self):
        """Test logger setup with default settings."""
        log = logger.setup_logger('default_test')
        
        self.assertIsInstance(log, logging.Logger)
        self.assertEqual(log.name, 'default_test')
        self.assertEqual(log.level, logging.INFO) # Default level
        self.assertEqual(len(log.handlers), 2) # Console and File handler by default
        
        # Check handler types (approximate check)
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in log.handlers))
        self.assertTrue(any(isinstance(h, logging.FileHandler) for h in log.handlers))
        
        # Check log file path for file handler
        file_handler = next((h for h in log.handlers if isinstance(h, logging.FileHandler)), None)
        self.assertIsNotNone(file_handler)
        self.assertEqual(file_handler.baseFilename, logger.DEFAULT_LOG_FILE)

    def test_setup_logger_custom_settings(self):
        """Test logger setup with custom level, file, and handlers."""
        log = logger.setup_logger(
            name='custom_test',
            log_level='DEBUG',
            log_file=self.log_file_path,
            use_console=False,
            use_file=True
        )
        
        self.assertEqual(log.name, 'custom_test')
        self.assertEqual(log.level, logging.DEBUG)
        self.assertEqual(len(log.handlers), 1) # Only file handler
        
        file_handler = log.handlers[0]
        self.assertIsInstance(file_handler, logging.FileHandler)
        self.assertEqual(file_handler.baseFilename, self.log_file_path)

    def test_setup_logger_no_file(self):
        """Test logger setup without a file handler."""
        log = logger.setup_logger(name='console_only', use_file=False)
        self.assertEqual(len(log.handlers), 1)
        self.assertIsInstance(log.handlers[0], logging.StreamHandler)

    def test_setup_logger_no_console(self):
        """Test logger setup without a console handler."""
        log = logger.setup_logger(name='file_only', use_console=False, log_file=self.log_file_path)
        self.assertEqual(len(log.handlers), 1)
        self.assertIsInstance(log.handlers[0], logging.FileHandler)
        self.assertEqual(log.handlers[0].baseFilename, self.log_file_path)

    def test_setup_logger_invalid_level(self):
        """Test logger setup with an invalid level string."""
        log = logger.setup_logger(name='invalid_level_test', log_level='INVALID')
        # Should default to INFO and print a warning
        self.assertEqual(log.level, logging.INFO)

    def test_logger_singleton(self):
        """Test that getting a logger with the same name returns the same instance."""
        log1 = logger.setup_logger('singleton_test')
        log2 = logger.setup_logger('singleton_test')
        self.assertIs(log1, log2)
        # Ensure handlers are not duplicated
        self.assertEqual(len(log1.handlers), 2) 
        self.assertEqual(len(log2.handlers), 2)

    def test_logging_output_file(self):
        """Test if messages are actually written to the log file."""
        log = logger.setup_logger(name='file_output_test', log_level='DEBUG', log_file=self.log_file_path, use_console=False)
        test_message_debug = f"Debug message {time.time()}"
        test_message_info = f"Info message {time.time()}"
        
        log.debug(test_message_debug)
        log.info(test_message_info)

        # Important: Close handlers to flush buffers before reading file
        for handler in log.handlers[:]:
             handler.close()
             log.removeHandler(handler) # Remove to avoid issues in tearDown

        self.assertTrue(os.path.exists(self.log_file_path))
        with open(self.log_file_path, 'r') as f:
            content = f.read()
            self.assertIn(test_message_debug, content)
            self.assertIn(test_message_info, content)
            self.assertIn('DEBUG', content)
            self.assertIn('INFO', content)


if __name__ == '__main__':
    unittest.main()