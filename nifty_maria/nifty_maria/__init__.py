import importlib_metadata
"""
nifty_maria.

Library for Nifty fits of maria data.
"""

meta = importlib_metadata.metadata('nifty_maria') 
__version__ = meta['version']
__author__ = meta['author-email']
__credits__ = 'TUM/CERN'
