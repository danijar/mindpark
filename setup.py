import sys
import os
import subprocess
import setuptools


PACKAGES = ['vizbot']
SETUP_REQUIRES = ['Sphinx']
INSTALL_REQUIRES = ['numpy', 'matplotlib', 'gym', 'sqlalchemy']


class Command(setuptools.Command):

    requires = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._returncode = 0

    def finalize_options(self):
        pass

    def run(self):
        if type(self).requires:
            self.distribution.fetch_build_eggs(type(self).requires)
            self.run_command('egg_info')
            self.reinitialize_command('build_ext', inplace=1)
            self.run_command('build_ext')
        self.__call__()
        if self._returncode:
            sys.exit(self._returncode)

    def call(self, command):
        env = os.environ.copy()
        env['PYTHONPATH'] = ''.join(':' + x for x in sys.path)
        self.announce('Run command: {}'.format(command), level=2)
        try:
            subprocess.check_call(command.split(), env=env)
        except subprocess.CalledProcessError as error:
            self._returncode = 1
            message = 'Command failed with exit code {}'
            message = message.format(error.returncode)
            self.announce(message, level=2)


class TestCommand(Command):

    requires = ['pytest', 'pytest-cov'] + INSTALL_REQUIRES
    description = 'run tests and create a coverage report'
    user_options = [
        ('target=', None, 'test package or file'),
        ('args=', None, 'args to forward to pytest')]

    def initialize_options(self):
        self.target = 'test'
        self.args = ''

    def __call__(self):
        command = 'python3 -m pytest --cov={} {} {}'
        packages = ','.join(PACKAGES)
        self.call(command.format(packages, self.target, self.args))


class LintCommand(Command):

    requires = ['pep8', 'pylint'] + INSTALL_REQUIRES
    description = 'run linters'
    user_options = [('args=', None, 'args to forward to pylint')]

    def initialize_options(self):
        self.args = ''

    def finalize_options(self):
        self.args += ' --rcfile=setup.cfg'

    def __call__(self):
        packages = ' '.join(PACKAGES)
        self.call('python3 -m pep8 {} test setup.py'.format(packages))
        for package in PACKAGES:
            self.call('python3 -m pylint {} {}'.format(package, self.args))


class FormatCommand(Command):

    requires = ['yapf']
    description = 'auto-format python code'
    user_options = []

    def initialize_options(self):
        self.args = ''

    def __call__(self):
        command = 'python3 -m yapf -r -i {} test setup.py'
        packages = ' '.join(PACKAGES)
        self.call(command.format(packages))


class SphinxCommand(Command):

    requires = ['Sphinx']
    description = 'create RST files and build sphinx docs using these files'
    user_options = []

    def initialize_options(self):
        self.args = ''

    def __call__(self):
        self.call('sphinx-apidoc -F -o doc/source ' + PACKAGES[0])
        self.call('python3 setup.py build_sphinx')


setuptools.setup(
    name=PACKAGES[0],
    version='0.2.0',
    description='Testbed for deep reinforcement learning algorithms',
    url='http://github.com/danijar/vizbot',
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    cmdclass={
        'test': TestCommand,
        'lint': LintCommand,
        'format': FormatCommand,
        'docs': SphinxCommand,
    })
