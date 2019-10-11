
from distutils.core import setup

setup(name='the_wizz',
      version="0.1.0",
      author="Christopher B. Morrison",
      author_email="morriscb@uw.edu",
      url="https://github.com/morriscb/the-wizz",
      packages=['the_wizz'],
      description="",
      long_description=open("README.md").read(),
      package_data={"": ["README.md", "LICENSE"],
                    "notebooks": ["notebooks/*"]
                    "the_wizz": ["data/*"]},
      include_package_data=True,
      install_requires=["astropy",
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "scipy",],
      classifiers=["Development Status :: 3 - Alpha",
                   "License :: OSI Approved :: Apache Software License",
                   "Intended Audience :: Science/Research",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python"],
      python_requires='>=3.6')
