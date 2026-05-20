from setuptools import find_packages, setup

package_name = 'railgate'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    include_package_data=True,
    package_data={'railgate': ['templates/*.html']},
    install_requires=['setuptools', 'flask'],
    zip_safe=True,
    maintainer='tamim',
    maintainer_email='jarifborno@gmail.com',
    description='Web UI for railgate crossing-status system.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'web_ui = railgate.web_ui:main',
            'sim_web_ui = railgate.sim_web_ui:main',
        ],
    },
)
