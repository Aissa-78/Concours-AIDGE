from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from install_platform import installation_plan


class PlatformInstallTests(unittest.TestCase):
    def test_linux_environments_do_not_reuse_windows_or_mac_venvs(self):
        folders = [installation_plan(name, 'Linux')[0].name for name in ('app', 'aidge', 'hardware')]
        self.assertEqual(len(set(folders)), 3)
        self.assertNotIn('.venv', folders)
        self.assertNotIn('.venv-aidge-colibry', folders)

    def test_windows_camera_has_its_own_python_and_no_driver_command(self):
        folder, python, commands = installation_plan('hardware', 'Windows')
        self.assertEqual(python.relative_to(folder).as_posix(), 'Scripts/python.exe')
        self.assertTrue(any('requirements-colibry-hardware.txt' in str(arg) for c in commands for arg in c))
        self.assertFalse(any(c[0].lower() in ('sudo', 'zadig', 'pnputil') for c in commands))

    def test_native_windows_aidge_routes_to_wsl(self):
        with self.assertRaisesRegex(ValueError, 'WSL'):
            installation_plan('aidge', 'Windows')

    def test_aidge_installer_requires_forward_backward_check(self):
        _, _, commands = installation_plan('aidge', 'Linux')
        self.assertTrue(any('check_aidge_training.py' in str(arg) for c in commands for arg in c))

    def test_invalid_platform_cannot_change_mac_environments(self):
        with self.assertRaises(ValueError):
            installation_plan('app', 'Darwin')


if __name__ == '__main__':
    unittest.main()
