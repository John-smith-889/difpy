import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(

    # name of the package
    name="difpy",
    version="0.0.1",
    author="John Smith",
    author_email="everyday.normal.hacker@gmail.com",
    description="Python package for information diffusion investigation in social networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/John-smith-889/difpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ] ,
###########################
# Dependencies management #
###########################
# # installation of required packages:
	install_requires=[
        'numpy',
	'networkx' #,
#    ],
#
# # Installation of additional packages from custom dependency links:
#    dependency_links=[
#        'git+http://github.com/user/repo/pkg_repo_with_init_and_stuff/master#egg=package-0.0.1',
#    ]
)

# https://python-packaging.readthedocs.io/en/latest/dependencies.html