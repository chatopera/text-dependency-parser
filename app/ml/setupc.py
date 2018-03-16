from distutils.core import setup, Extension

module1 = Extension('ml',
                    sources=['ml.c'])

setup(name='ml',
      version='1.0',
      ext_modules=[module1])
