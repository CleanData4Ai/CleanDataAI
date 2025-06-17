from setuptools import setup, find_packages

setup(
    name='data_analysis_exploration',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'beautifulsoup4==4.11.1',
        'fastparquet==2024.5.0',
        'ipython==8.14.0',
        'ipython-genutils==0.2.0',
        'matplotlib==3.7.0',
        'matplotlib-inline==0.1.6',
        'mpld3==0.5.10',
        'numpy==1.23.5',
        'pandas==2.2.2',
        'pyarrow==17.0.0',
        'pyarrow-hotfix==0.5',
        'scipy==1.10.0',
        'seaborn==0.12.2',
        'tabulate==0.9.0',
        'ydata-profiling==4.10.0',
        'setuptools',
        'wheel',
        'twine'
    ],
    author='Mohamed Naceur Mahmoud',
    author_email='mohamednaceurmahmoud98@gmail.com',
    description='This is a Python package that automates the process of exploratory data analysis (EDA). '
                'It can perform a variety of EDA tasks, making it easier for '
                'users to gain insights from their data quickly and efficiently.',
)
